## VTool-R1 New Training Library

This library contains the new training and evaluation code for VTool-R1.

It is implemented using the latest version of verl, commit 498c988ab7af49aa36c157c9214ebbc780013d61, using VLLM 0.17

The library is implemented using the asynchronous agent loop, allowing for max GPU utilization for mult-turn tool use.

### Dataset

All dataset available are on our hugging face page. You may find it [here](https://huggingface.co/datasets/VTOOL/Refocus_Chart)

## Training

We implemented VTool-R1 in the form of a recipe, available in recipe/vtool

Training scripts for 3B and 7B are available 

~~~
bash run_qwen2_5_vl_3b_chart.sh

bash run_qwen2_5_vl_7b_chart.sh
~~~

The scripts are configured for a single 4xH100 node.

As stated in our original paper, we require an additional LLM serving as the reward reward. We provide a much more efficient strategy for doing so.

By default, the training script will attempt to connect to your vllm serve endpoint on localhost:8000/v1, you may configure it to use another GPU node.

You may either take advantage of your CPUs to host the LLM judge, or partition a small chunk of your GPU memory to host the LLM judge. 

In addition, we provide hybrid solutions in the form of the load_balance.sh. You can then run both your CPU judge, as well as the small GPU judge and route them through the load balancer. Our ratio was tested for optimal performance on a GH100 node (4xH100).

## Evaluation

Evaluation scripts are in the eval folder.

Please launch vllm_infer.sh to boot up the target model for evaluation using vllm serve. Then, use run_eval.sh to complete the evaluation rollout.

Then, you should use score_results.sh to score the results. You may use your own openai api key or GPT OSS 120B through vllm.
