+++
title = 'Arabic RAG 1: Getting the Data'
date = 2023-11-26T07:32:41+04:00
author = "Derek Thomas"
draft = false
ShowReadingTime = true
tags = ["Arabic NLP", "Arabic RAG", "Tutorial"]
cover.image = "cover_images/arabic-rag-1.png"
cover.alt = "Photo Credits to DALL·E 3"
+++

# Goal
This is part 1 of 6 on a tutorial for **Arabic RAG**. RAG is short for Retrieval Augmented Generation. It took it's name
from [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) though it's 
current usage is much more similar to the [RALM Paper](https://arxiv.org/abs/2302.00083). 

In this tutorial you will learn:
- Why is RAG important
- How to download Wikipedia
- How to format Wikipedia for scalable processing

## Addressing Hallucinations
Large Language Models (LLMs) get blamed (though unfairly IMHO) for "hallucinating". Often we anthropomorphize LLMs 
[1](https://arxiv.org/pdf/2212.03551.pdf) like they are a competant knowledge base when in fact they don't really 
"*know*" things like humans do. The reality is that they are just good at predicting the next token (similar to a word) 
in a sequence of text. 

## How does RAG work
RAG is a way of conditioning the prompt with grounded truth. We retrieve relevant documents and 
ask the LLM to answer questions only based on the information in your documents. This is a much easier problem and we 
have seen great results and much fewer "hallucinations".

## Tutorial Notes

I'll be downloading and processing 
[Arabic Wikipedia](https://ar.wikipedia.org/wiki/%D8%A7%D9%84%D8%B5%D9%81%D8%AD%D8%A9_%D8%A7%D9%84%D8%B1%D8%A6%D9%8A%D8%B3%D9%8A%D8%A9).
as a knowledge base for RAG. I chose this as it's a great source of knowledge on a variety of tasks. With millions of 
articles, it will also force us to use best practice for scale which is missing in many tutorials in general. Using 
Arabic Wikipedia is a best case scenario source of information since all (or most) LLMs are pre-trained on Wikipedia. 
I'll be using [jais-13B](https://huggingface.co/core42/jais-13b-chat) from core42 as the LLM. Check out the 
[paper](https://huggingface.co/papers/2308.16149) here to see the training details.

{{< notice note >}}
There are a couple sticking points in this tutorial since we need a **directory structure** and
**supporting files** and these don't easily translate in a tutorial.
{{< /notice >}}
{{< notice warning >}}
The easiest way to run this would be to:

1. `git clone https://huggingface.co/spaces/derek-thomas/arabic-RAG`
1. `cd arabic-RAG`
1. `jupyter lab`
1. Use this as a guide.
{{< /notice >}}


# Get Data
The data from wikipedia starts in XML, our approach will be converting this into a series of `.ndjson` files for easy
processing.

If you have any ideas on how this could be improved please do let me know in the comments!

## Initialize
```python
from pathlib import Path
import sys
```

```python
proj_dir_path = Path.cwd().parent
proj_dir = str(proj_dir_path)

# So we can import later
sys.path.append(proj_dir)
```

## Install

{{< notice note >}}
This is the first sticking point. You can find the file [here](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/requirements.txt)
{{< /notice >}}

```jupyter
%pip install -q -r "$proj_dir"/requirements.txt
```

    Note: you may need to restart the kernel to use updated packages.

## Download Arabic Wikipedia

{{< notice tip >}}
Im getting "latest" but it's good practice to see what version it is.
{{< /notice>}}

```jupyter
!curl -I https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles-multistream.xml.bz2 --silent | grep "Last-Modified"
```

    Last-Modified: Sat, 21 Oct 2023 02:57:42 GMT

Download simple wikipedia.

{{< notice note >}}
Note that there is an implied directory structure here. And you will be putting your data in
`../data` for this and the subsequent tutorials in this series.
{{< /notice >}}

```jupyter
!wget -nc -P "$proj_dir"/data/raw https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles-multistream.xml.bz2
```

    --2023-10-28 08:09:45--  https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles-multistream.xml.bz2
    Resolving dumps.wikimedia.org (dumps.wikimedia.org)... 208.80.154.142, 2620:0:861:2:208:80:154:142
    Connecting to dumps.wikimedia.org (dumps.wikimedia.org)|208.80.154.142|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1671369109 (1.6G) [application/octet-stream]
    Saving to: ‘/home/ec2-user/arabic-wiki/data/raw/arwiki-latest-pages-articles-multistream.xml.bz2’
    
    100%[====================================>] 1,671,369,109 4.54MB/s   in 5m 54s 
    
    2023-10-28 08:15:39 (4.51 MB/s) - ‘/home/ec2-user/arabic-wiki/data/raw/arwiki-latest-pages-articles-multistream.xml.bz2’ saved [1671369109/1671369109]

## Extract

The download format from wikipedia is in XML. `wikiextractor` will convert this into a jsonl format split into many
folders and files.

```jupyter
!wikiextractor -o "$proj_dir"/data/raw/output  --json "$proj_dir"/data/raw/arwiki-latest-pages-articles-multistream.xml.bz2 
```

    INFO: Preprocessing '/home/ec2-user/arabic-wiki/data/raw/arwiki-latest-pages-articles-multistream.xml.bz2' to collect template definitions: this may take some time.
    INFO: Preprocessed 100000 pages
    ...
    INFO: Extracted 2200000 articles (2416.5 art/s)
    INFO: Finished 3-process extraction of 2254650 articles in 641.2s (3516.3 art/s)

## Consolidate

The split format is tedious to deal with, so now we we will consolidate this into a series of  `.ndjson` files. This is
important as our data processing machine might not have enough RAM to fit the whole dataset. 
I like `.ndjson` because it:
- Allows us to stream easily due to being able to split on rows/lines (regular `.json` is too messy to easily get the first `x rows`)
- Has higher fault tolerance: If one line is corrupted the file is not corrupted
- Facilitates parallel processing
- Allows incremental loading

Dive into the [consolidate functionality](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/src/preprocessing/consolidate.py) for more details.

{{< notice note >}}
This is the second sticking point, as it's not easy to place extra code. Here is
[consolidate.py](https://huggingface.co/spaces/derek-thomas/arabic-RAG/blob/main/src/preprocessing/consolidate.py) in
case you aren't using the repository.
{{< /notice >}}

```python
from src.preprocessing.consolidate import folder_to_json
```

```python
folder = proj_dir_path / 'data/raw/output'
folder_out = proj_dir_path / 'data/consolidated/'
folder_to_json(folder, folder_out, 'ar_wiki')
```

    Processing: 100%|█████████████████| 6119/6119 [01:11<00:00, 85.38file/s, File: wiki_18 | Dir: /home/ec2-user/arabic-wiki/data/raw/output/CJ]

    Wiki processed in 72.87 seconds!
    
We did it! We now have the latest Arabic Wikipedia in a digestable format. Lets do some analysis in Part 2 to understand
how we should chunk our data.

# Tutorial Design Decisions
I structured this in a repo for a couple reasons:
- It allows good software practices like abstraction
- It is easier to create a data system that will work for the future tutorials
- Its cleaner and easier to maintain

It does pose a challenge when I describe my work. Many times readers would rather just click a button and rush through
a notebook. Thats fair, I've been there. In this case I wanted to take a deeper more intentional look. Feel free to
comment with your thoughts!