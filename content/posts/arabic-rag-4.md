+++
title = 'Arabic RAG 4: Get Embeddings'
date = 2023-12-03T17:13:34+04:00
author = "Derek Thomas"
draft = false
ShowReadingTime = true
tags = ["Arabic NLP", "Arabic RAG", "Tutorial"]
cover.image = "cover_images/arabic-rag-4.png"
cover.alt = "Photo Credits to DALL·E 3"
+++

# Goals
This is part 4 of 6 in our tutorial on Arabic RAG. I'll be using this blog as a guide, but to actually run this
tutorial, its best that you run
[this notebook](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/notebooks/04_get_embeddings.ipynb)
as described in [part 1]({{< ref "arabic-rag-1" >}}).

Before diving in, I want to you to think about how much money it costs to embed 2M articles. Make an estimate and see
how accurate your guess is. You can find out what other people thought in 
[this poll](https://www.linkedin.com/feed/update/urn:li:ugcPost:7130532349741514752/) I conducted 
(*spoiler in the comments*).

{{< notice info >}}
In this blog you will learn:
- What the challenges are with embedding for RAG at scale
- How to solve this quickly using [TEI](https://github.com/huggingface/text-embeddings-inference) and 
[Inference Endpoints](https://huggingface.co/inference-endpoints)
- How to embed your chunks in a cost-effective manner
{{< /notice >}}

Getting your embeddings isn't talked about enough. It's always surprisingly difficult as you scale and most tutorials 
aren't at any real scale. The right answer used to be 
[exporting to ONNX](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) with `O4` level 
optimization and running it from there. While not difficult it took a little know-how and some preparation. But there 
have been a lot more developments lately.

## Why TEI
The problem with getting the embeddings efficiently is that there are techniques that exist but they are not widely 
*applied*. [TEI](https://github.com/huggingface/text-embeddings-inference#docker) solves a number of these:
- Token Based Dynamic Batching
- Using latest optimizations ([Flash Attention](https://github.com/HazyResearch/flash-attention), 
[Candle](https://github.com/huggingface/candle) and
[cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api))
- Fast and safe loading with [safetensors](https://github.com/huggingface/safetensors)
- Parallel Tokenization workers

Applying most of these is doable but quite tedious. Big thanks to [Olivier](https://github.com/OlivierDehaene) for 
creating this!

# Set up TEI
There are 2 ways you can go about running [TEI](https://github.com/huggingface/text-embeddings-inference#docker), 
locally or with [Inference Endpoints](https://huggingface.co/inference-endpoints). Given not everyone will want to use 
[Inference Endpoints](https://huggingface.co/inference-endpoints) as it is paid I have instructions on how to do this 
locally to be more inclusive!

## Start TEI Locally
I have this running in a nvidia-docker container, but there are 
[other ways to install too](https://github.com/huggingface/text-embeddings-inference#local-install). Note that I ran 
the following docker startup cell in a different terminal for monitoring and separation. 

{{< notice note >}}
The way I have written the docker command will pull the latest image each time its run. TEI is at a very early stage at 
the time of writing. You may want to pin this to a specific version if you want more repeatability. 
{{< /notice >}}

As described in [part 3]({{< ref "arabic-rag-3" >}}), I chose 
[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) based on the STS ar-ar performance on 
[mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard), it's the top performer and half the size of second 
place! TEI is fast, but this will also make our life easy for storage and retrieval.

I use the `revision=refs/pr/8` because this has the pull request with [safetensors](https://github.com/huggingface/safetensors) which is required by TEI. Check
out the [pull request](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/discussions/8) if you want to use a different embedding model and it doesnt have safetensors.
```bash
echo "Copy and paste this in another terminal. Make sure you have installed nvidia-docker and build as described here:
https://github.com/huggingface/text-embeddings-inference#docker-build"

volume=$pwd/tei
model=sentence-transformers/paraphrase-multilingual-minilm-l12-v2
revision=refs/pr/8
docker run \
    --gpus all \
    -p 8080:80 \
    -v $volume:/data \
    -v /home/ec2-user/.cache/huggingface/token:/root/.cache/huggingface/token \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id $model \
    --revision $revision \
    --pooling mean \
    --max-batch-tokens 65536
```

### Test Endpoint


```bash
echo "This is just a sanity check so you don't get burned later."
response_code=$(curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json')

if [ "$response_code" -eq 200 ]; then
    echo "passed"
else
    echo "failed"
fi
```

{{< notice warning >}}
I'll be running the rest of the notebook with Inference Endpoints. If you are using the local version, you will need to 
make some minor code updates to point to the right endpoint.
{{< /notice >}}

## Start TEI with Inference Endpoints
Another option is to run TEI on [Inference Endpoints](https://huggingface.co/inference-endpoints). Its cheap and fast. It took me less than 5 minutes to get it
up and running!

Check here for a [comprehensive guide](https://huggingface.co/blog/inference-endpoints-embeddings#3-deploy-embedding-model-as-inference-endpoint). For our tutorial make sure to set these options **IN ORDER**:
1. Model Repository = `transformers/paraphrase-multilingual-minilm-l12-v2`
1. Name your endpoint `TEI-is-the-best`
1. Choose a GPU, I chose `Nvidia A10G` which is **$1.3/hr**.
1. Advanced Configuration
    1. Task = `Sentence Embeddings`
    1. Revision (based on [this pull request for safetensors](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/discussions/8) = `a21e6630`
    1. Container Type = `Text Embeddings Inference`
    
Set the other options as you prefer.

### Test Endpoint
I chose to keep these hidden because its sensitive, [getpass](https://docs.python.org/3/library/getpass.html) is a 
great way to hide sensitive data entry for public facing code. 
```python
import getpass
API_URL = getpass.getpass(prompt='What is your API_URL?')
bearer_token = getpass.getpass(prompt='What is your BEARER TOKEN? Check your endpoint.')
```
    What is your API_URL? ········
    What is your BEARER TOKEN? Check your endpoint. ········

Here we have a couple constants. The headers for our request and our `MAX_WORKERS` for parallel inference. 
```python
# Constants
HEADERS = {
	"Authorization": f"Bearer {bearer_token}",
	"Content-Type": "application/json"
}
MAX_WORKERS = 512
```

Let's make sure everything is running correctly. Error messages with 512 workers is no fun.
```python
import requests

def query(payload):
	response = requests.post(API_URL, headers=HEADERS, json=payload)
	return response.json()
	
output = query({
	"inputs": "This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music!",
})
print(f'{output[0][:5]}...')
```
    [0.0047912598, -0.03164673, -0.018051147, -0.057739258, -0.04498291]...

Ok great! We have TEI up and running!

# Get Embeddings

## Imports
```python
import asyncio
from pathlib import Path
import json
import time


from aiohttp import ClientSession, ClientTimeout
from tqdm.notebook import tqdm
```

```python
proj_dir = Path.cwd().parent
print(proj_dir)
```
    /home/ec2-user/arabic-wiki

## Config

As mentioned before I think it's always a good idea to have a central place where you can parameterize/configure your notebook.
```python
files_in = list((proj_dir / 'data/processed/').glob('*.ndjson'))
folder_out = proj_dir / 'data/embedded/'
folder_out_str = str(folder_out)
```

## Strategy
TEI allows multiple concurrent requests, so its important that we dont waste the potential we have. I used the default 
`max-concurrent-requests` value of `512`, so I want to use that many `MAX_WORKERS`. This way we can maximize the 
utilization of the GPU.

Im using an `async` way of making requests that uses `aiohttp` as well as a nice progress bar. What better way to do 
Arabic NLP than to use [tqdm](https://github.com/tqdm/tqdm) which has an Arabic name?

Note that Im using `'truncate':True` as even with our `225` word split earlier, there are always exceptions. For some 
use-cases this wouldn't be the right answer as throwing away information our retriever could use can lower performance
and cause insidious error propogation. You can pre-tokenize, assess token count, and re-chunk as needed.

This is our code to efficiently leverage TEI.
```python
async def request(document, semaphore):
    # Semaphore guard
    async with semaphore:
        payload = {
            "inputs": document['content'],
            "truncate": True
        }
        
        timeout = ClientTimeout(total=10)  # Set a timeout for requests (10 seconds here)

        async with ClientSession(timeout=timeout, headers=HEADERS) as session:
            async with session.post(API_URL, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(await resp.text())
                result = await resp.json()
                
        document['embedding'] = result[0]  # Assuming the API's output can be directly assigned
        return document

async def main(documents):
    # Semaphore to limit concurrent requests. Adjust the number as needed.
    semaphore = asyncio.BoundedSemaphore(512)

    # Creating a list of tasks
    tasks = [request(document, semaphore) for document in documents]
    
    # Using tqdm to show progress. It's been integrated into the async loop.
    for f in tqdm(asyncio.as_completed(tasks), total=len(documents)):
        await f
```

Now that we have all the pieces in place we can get embeddings. Our high level approach:
1. Read each processed `.ndjson` chunk file into memory
2. Use parallel workers to:
   1. Get embeddings for each `document`
   2. Update each `document` with the corresponding embedding
3. Verify that we got embeddings (always error check)
4. Write these to file.
```python
start = time.perf_counter()
for i, file_in in tqdm(enumerate(files_in)):

    with open(file_in, 'r') as f:
        documents = [json.loads(line) for line in f]
        
    # Get embeddings
    await main(documents)
        
    # Make sure we got it all
    count = 0
    for document in documents:
        if document['embedding'] and len(document['embedding']) == 384:
            count += 1
    print(f'Batch {i+1}: Embeddings = {count} documents = {len(documents)}')

    # Write to file
    with open(folder_out/file_in.name, 'w', encoding='utf-8') as f:
        for document in documents:
            json_str = json.dumps(document, ensure_ascii=False)
            f.write(json_str + '\n')
            
# Print elapsed time
elapsed_time = time.perf_counter() - start
minutes, seconds = divmod(elapsed_time, 60)
print(f"{int(minutes)} min {seconds:.2f} sec")
```


    0it [00:00, ?it/s]
      0%|          | 0/243068 [00:00<?, ?it/s]
    Batch 1: Embeddings = 243068 documents = 243068
      ...
      0%|          | 0/70322 [00:00<?, ?it/s]
    Batch 23: Embeddings = 70322 documents = 70322
    104 min 32.33 sec

Wow, it only took ~1hr 45 min!

Lets make sure that we have all our documents:
```python
!echo "$folder_out_str" && cat "$folder_out_str"/*.ndjson | wc -l
```

    /home/ec2-user/arabic-wiki/data/embedded
    2094596
Great, it looks like everythign is correct.


# Performance and Cost Analysis
You can see that we are quite cost effective! For only $2.30 we got text embeddings for ~2M articles!! How does that
compare with your expectations?

![Cost](https://huggingface.co/spaces/derek-thomas/arabic-RAG/resolve/main/media/arabic-rag-embeddings-cost.png)

Note that the performance shown is over just the last 30 min window.
Observations:
- We have a througput of `~334/s`
- Our median latency per request is `~50ms`

![Metrics](https://huggingface.co/spaces/derek-thomas/arabic-RAG/resolve/main/media/arabic-rag-embeddings-metrics.png)

# Next Steps
In the next blogpost I'll show an exciting solution on VectorDB choice and usage. I was blown away since I expected it 
to take about a day to do the full ingestion, but it was MUCH FASTER 😱. 
