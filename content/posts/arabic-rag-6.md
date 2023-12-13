+++
title = 'Arabic RAG 6: Putting it together'
date = 2023-12-13T11:53:29+04:00
author = "Derek Thomas"
draft = false
ShowReadingTime = true
tags = ["Arabic NLP", "Arabic RAG", "Tutorial"]
cover.image = "cover_images/arabic-rag-6.png"
cover.alt = "Photo Credits to DALL·E 3"
+++

# Goal

This is part 6 of 6 in our tutorial on Arabic RAG. We have created all of the components we need to make our RAG
solution. All that is left is to stitch them together!

{{< notice note >}}
In this blog you will learn how to:

- Quickly and efficiently deploy `jais` using Inference Endpoints
- Combine all the components of RAG into a functional system
- Create a beautiful Gradio App for RAG
{{< /notice >}}

**If you want to skip all this and actually try the app here it is**: https://huggingface.co/spaces/derek-thomas/arabic-RAG

{{< notice warning >}}
1. Click Wake Up Endpoint if its `scaledToZero`
2. I'm a bit nervous how this will handle traffic as this is just a demo app. Be patient and considerate please 🙏🏾
{{< /notice >}}

I do hope to do some work on using [jais](https://huggingface.co/core42/jais-13b-chat) for high throughput in the future :) 

# Handling Jais

First lets understand a couple of things about [jais](https://huggingface.co/core42/jais-13b-chat). It is a 13B model so our options are going to be limited on where
we can deploy it. Another important point is while it is on the [huggingface hub](https://huggingface.co) it uses remote
code `trust_remote_code=True`. This just means it isn't natively integrated with the
[transformers library](https://huggingface.co/docs/transformers/index). It uses extra code to handle the architecture.
This is common when developing new architectures. Where this becomes relevant is it can be tricking
working with [TGI](https://huggingface.co/docs/text-generation-inference/index) or
[Inference Endpoints](https://huggingface.co/inference-endpoints).

To use [jais](https://huggingface.co/core42/jais-13b-chat) for our arabic-rag project we need to serve it somewhere. I chose
[Inference Endpoints](https://huggingface.co/inference-endpoints) as it allows me to chose whatever Hardware I want from
AWS or Azure.

## Jais on Inference Endpoints

Jais is easy to deploy on Inference Endpoints, all you need to do is click deploy then select `Inference Endpoints`.

There are a few considerations on we need to handle to use jais for RAG before we do that:

- Jais as-is requires expensive GPU configurations
- We need to be able to choose a system prompt targeted for RAG

Inference Endpoints gives you [an option](https://huggingface.co/docs/inference-endpoints/guides/custom_handler) to
create a `handler.py` file where you specify:

- How to load your model in `__init__`
- How do call your model in `__call__`

You can see a couple key differences in the
[original handler](https://huggingface.co/core42/jais-13b-chat/blob/main/handler.py)
and [my handler](https://huggingface.co/derek-thomas/jais-13b-chat-hf/blob/main/handler.py)

### Jais VRAM optimization

13B means that I will need ~52GB of VRAM just for the weights at full-precision. There arent many GPUs which can handle
that. This doesn't consider activations and the quadratic attention which will grow quickly at the 2k context length.
Luckily we are able to use LLM.int8()'s ([paper](https://huggingface.co/papers/2208.07339),
[tutorial](https://huggingface.co/blog/hf-bitsandbytes-integration)) implementation in
[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) since jais is compatible with
[accelerate](https://huggingface.co/docs/accelerate/index). The tl;dr is that this allows us to use jais with no
decrease in performance in 8 bits! That means that we only need 13GB for the weights.

When I load my model I use `load_in_8bit=True` which allow the model to utilize LLM.int8().

```python
self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto",
                                                  offload_folder='offload',
                                                  trust_remote_code=True,
                                                  load_in_8bit=True)
```

### System Prompting Jais in Inference Endpoints

When I call my model I allow the user to specify the full prompt based on the API call to Inference Endpoints. You can
also use the default system prompt by calling it normally.

```python
# Give the user the opportunity to override the prompt
if 'prompt' in data.keys():
    text = data['prompt']
```

I made these changes in [derek-thomas/jais-13b-chat-hf](https://huggingface.co/derek-thomas/jais-13b-chat-hf) as well as
merging safetensors. But feel free to explore this and make your own `handler.py`.


# Code Explanations

Great, now we have all our pieces in place we need to build an app that uses all of this! To see the full app check out
[the code](https://huggingface.co/spaces/derek-thomas/arabic-RAG/tree/main). But I will be focusing on key snippets on
how to create the app. I know not everyone cares, so I minimized it by default

I hid some analysis below since it was getting quite long. But its worth a dive if you are going to implement it.

<details>
  <summary><b>Click if you want to see some detailed code analysis!</b></summary>

## Pre-requisites

### Loading LanceDB

Its super easy to load LanceDB:

```python
# Start the timer for loading the VectorDB
start_time = time.perf_counter()

proj_dir = Path(__file__).parents[1]

# Log the time taken to load the VectorDB
db = lancedb.connect(proj_dir / "lancedb")
tbl = db.open_table('arabic-wiki')
lancedb_loading_time = time.perf_counter() - start_time
logger.info(f"Time taken to load LanceDB: {lancedb_loading_time:.6f} seconds")
```

> INFO:backend.semantic_search:Time taken to load LanceDB: 0.000174 seconds

It's file-based so we would expect this to take a while, but the wizards at LanceDB give us greatness for free.

### Load Embedding Model

```python
st_model_cpu = SentenceTransformer(name, device='cpu')
```

## Embed Query

LanceDB is a bit bare bones at the moment. We need to write a simple retriever function. No worries its super easy:

```python
def vector_search(query_vector, top_k):
    return tbl.search(query_vector).limit(top_k).to_list()


def retriever(query, top_k=3):
    query_vector = call_embed_func(query)
    documents = vector_search(query_vector, top_k)
    return documents
```

Great now we have the pieces to retrieve documents based on a query. Next lets query jais from our inference endpoint.

## jais Handling

Here we construct the format expected by our custom `handler.py` Inference Endpoints. Now we can send a special RAG
prompt to jais.

```python
def generate(prompt: str):
    start_time = time.perf_counter()

    payload = {'inputs': '', 'prompt': prompt}
    response = call_jais(payload)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.warning(f"Function took {elapsed_time:.1f} seconds to execute")

    return response
```

## App Creation

Hopefully the details of [app.py](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/app.py) are clear. If 
you have any questions I didn't address let me know in the comments below. 

The basic idea is we use the `gr.Chatbot` component to do the heavy lifting. We don't want users sending in requests
while we are processing their current request so we turn it off for both `txt_btn.click` and `txt.submit`:

```python
# Turn off interactivity while generating if you click
txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, [chatbot, prompt_html])

# Turn it back on
txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

# Turn off interactivity while generating if you hit enter
txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, [chatbot, prompt_html])

# Turn it back on
txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
```

</details>

## RAG Prompt
I'm using a simple [jinja](https://jinja.palletsprojects.com/en/3.1.x/) template
[template.j2](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/templates/template.j2) to handle the
interpolation from our prompt, queried documents, and our query.

```jinja2
### Instruction: استخدم المستندات الفريدة التالية في قسم السياق للإجابة على الاستعلام في النهاية. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف، ولا تحاول اختلاق إجابة.
### Context
{% for doc in documents %}
---
    {{ doc.content }}
{% endfor %}
---
[|AI|]:
### Query: [|Human|] {{query}}
### Response: [|AI|]
```
Ultimately this is what gets sent to jais and the response is fed back to the chatbot. Here is a visualized version that
we will see in the application itself. 
![prompt_screenshot.png](/posts/arabic-rag-6/prompt_screenshot.png)

# Conclusion
Thanks so much for following along! I hope this was helpful. I know 6 parts is a lot, but I could have gone much deeper
and made this 12 parts haha. If you have any comments please leave them below. 

I do plan on writing more on jais, specifically fine-tuning best practices, I want to evaluate its instruction following 
ability, and possible LLM serving. If you have other ideas let me know!

If you did like this series please click like below or leave a comment below. It would mean a lot to me!
