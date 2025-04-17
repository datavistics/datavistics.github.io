+++
title = 'AI Watermarking 101: Tools and Techniques'
date = 2024-04-17T10:00:00+04:00
author = "Derek Thomas"
draft = false
ShowReadingTime = true
tags = ["AI", "Watermarking", "Deepfakes", "AI Safety"]
cover.image = "cover_images/ai-watermarking.png"
cover.alt = "AI Watermarking Concept"
+++

# Introduction

In recent months, we've seen numerous news stories involving 'deepfakes' - AI-generated content that can be incredibly convincing and potentially harmful. From fake images of Taylor Swift to fabricated videos of Tom Hanks and synthetic recordings of President Biden, these deepfakes spread rapidly on social media platforms, causing real damage before they can be debunked.

In this post, I'll summarize key points from [Hugging Face's excellent article on AI watermarking](https://huggingface.co/blog/watermarking), exploring how watermarking techniques can help identify AI-generated content and mitigate some of these risks.

{{< notice info >}}
This post is a summary of the original article published by Hugging Face. For the complete technical details, please refer to the [original publication](https://huggingface.co/blog/watermarking).
{{< /notice >}}

# What is AI Watermarking?

Watermarking in AI involves adding patterns to digital content (such as images, text, or audio) to convey information about its provenance. These patterns can range from fully visible (like OpenAI's DALL-E 2 colored blocks) to completely invisible, detectable only through specialized algorithms.

![Dall-E 2 watermark example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/watermarking/fig1.png)

There are two primary methods for watermarking AI-generated content:
1. **During content creation** - embedded as part of the generation process, requiring access to the model itself
2. **After content production** - applied to already generated content, which can work with closed-source models

# Watermarking Different Types of Data

## Image Watermarking

For images, several approaches exist:

- **Image Cloaking** tools like "Nightshade" and [Fawkes](https://huggingface.co/spaces/derek-thomas/fawkes) make tiny changes to images that are imperceptible to humans but impact models trained on this data
- **Output Watermarking** tools like [IMATAG](https://huggingface.co/spaces/imatag/stable-signature-bzh) and [Truepic](https://huggingface.co/spaces/Truepic/watermarked-content-credentials) add invisible watermarks during or after image generation

## Text Watermarking

Text watermarking is more challenging but equally important. Current approaches typically:

1. Split candidate tokens into two groups (often called "red" and "green")
2. Promote or restrict tokens based on previous generated text
3. Detection works by calculating the probability that input text comes from a specific model

The [Watermark for LLMs Space](https://huggingface.co/spaces/tomg-group-umd/lm-watermarking) demonstrates this approach, and Hugging Face's [Text Generation Inference toolkit](https://huggingface.co/docs/text-generation-inference/index) implements watermarking algorithms that can be used with the latest models.

{{< notice warning >}}
Text watermarking detection requires substantial text to be reliable. Even then, detectors can have high false positive rates, incorrectly labeling human-written text as synthetic.
{{< /notice >}}

## Audio Watermarking

Voice data is often used as a biometric security measure, making audio watermarking particularly important. Approaches like [AudioSeal](https://github.com/facebookresearch/audioseal) embed watermarks in frequencies imperceptible to human ears while maintaining robustness against editing.

# Limitations and Considerations

While watermarking is a promising approach for identifying AI-generated content, it's not foolproof:

- Watermarks can potentially be removed by determined actors
- Detection accuracy varies across modalities and techniques
- There's a balance between open and closed watermarking systems (transparency vs. security)

# Conclusion

As AI-generated content becomes increasingly sophisticated and widespread, watermarking provides an important tool for maintaining transparency about content origins. The Hugging Face Hub offers several tools for both applying and detecting watermarks across different modalities.

While not perfect, these techniques represent an important step toward responsible AI development and use. As the field evolves, we can expect more sophisticated watermarking approaches to emerge, helping us navigate the complex landscape of synthetic media.

# Further Resources

- [Hugging Face's original article on AI watermarking](https://huggingface.co/blog/watermarking)
- [IMATAG watermarking space](https://huggingface.co/spaces/imatag/stable-signature-bzh)
- [Truepic watermarking space](https://huggingface.co/spaces/Truepic/watermarked-content-credentials)
- [Watermark for LLMs Space](https://huggingface.co/spaces/tomg-group-umd/lm-watermarking)
