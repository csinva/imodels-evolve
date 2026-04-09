The goal is to evaluate how well AI agent systems can do at end-to-end data-science tasks.

We will evaluate on the [Blade](https://github.com/behavioral-data/blade) dataset, which contains 12 datasets, each with a data-science task and a corresponding solution.

The evaluation will be done in a zero-shot setting, where the system is given the task description and the dataset, and is asked to produce a solution. Do not use the scaffolding provided in the Blade repository, simply use the task description and the dataset as input to the AI agent system.

For the AI agent system, use OpenAI Codex using the Azure OpenAI with Entra ID Authentication as shown in the `example-blade-repo` folder. You may borrow some code from that folder for how to set up blade, but simplify things greatly as that folder contains a lot of unnecessary code for our purposes. The goal is to have a simple script that can run Codex on all 12 datasets, then compare the resulting analysis to the human-written solutions. Put all your code into a new folder titled `blade-evaluation`.

Finally, run the scripts and report the evaluation using LLM-as-a-judge to compare to the human solutions. Use the same Azure OpenAI setup for this as well. The evaluation should be based on how well the AI agent system's solution matches the human-written solution in terms of correctness, completeness, and clarity. Write a brief report summarizing the results of the evaluation.
