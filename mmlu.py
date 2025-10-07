from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.scorer import Scorer
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.telemetry import capture_benchmark_run
import warnings

class MMLU(DeepEvalBaseBenchmark):
    def __init__(
        self, tasks: List[MMLUTask] = None, n_shots: int = 5, **kwargs
    ):
        assert n_shots <= 5, "MMLU only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[MMLUTask] = list(MMLUTask) if tasks is None else tasks
        self.scorer = Scorer()
        self.shots_dataset: List[Dict] = None
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        
    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        with capture_benchmark_run("MMLU", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []
            use_batch = should_use_batch(model, batch_size)

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task)
                task_correct_predictions = 0
                task_total_predictions = len(goldens)
                overall_total_predictions += len(goldens)

                # Calculate task accuracy
                for golden in tqdm(
                    goldens, desc=f"Processing {task.value}"
                ):
                    prediction, score = self.predict(
                        model, task, golden
                    ).values()
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (task.value, golden.input, prediction, score)
                    )

                task_accuracy = (
                    task_correct_predictions / task_total_predictions
                )
                print(
                    f"MMLU Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall MMLU Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            # Columns: 'Task', 'Input', 'Prediction', 'Score'
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=["Task", "Input", "Prediction", "Correct"],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy
        
    def predict(
        self, model: DeepEvalBaseLLM, task: MMLUTask, golden: Golden
    ) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = MMLUTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
        )

        full_answer = model.generate(prompt)
        prediction = self.extract_answer(full_answer)

        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def get_single_pair(
        self, task: MMLUTask, golden: Golden
    ) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = MMLUTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
        )
        return {"prompt": prompt, "golden_label": golden.expected_output}
    
    def get_pairs(
        self, value_to_check=None
    ) -> Dict:
        with capture_benchmark_run("MMLU", len(self.tasks)):
            prompts = {}
            golden_labels = {}
            
            for task in self.tasks:
                prompts[task.value] = []
                golden_labels[task.value] = []
                goldens = self.load_benchmark_dataset(task)

                # Calculate task accuracy
                for golden in tqdm(
                    goldens[:], desc=f"Processing {task.value}"
                ):
                    prompt, golden_label = self.get_single_pair(
                        task, golden
                    ).values()
                    
                    prompts[task.value].append(prompt)
                    golden_labels[task.value].append(golden_label)
                    
            return prompts, golden_labels                
                
    def calc_accuracies(
        self, ques: Dict, full_answers: Dict, golden_labels: Dict, value_scores: Dict, value: str
    ) -> Dict:
        with capture_benchmark_run("MMLU", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []

            for task in full_answers.keys():
                ques_task = ques[task]
                full_answers_task = full_answers[task]
                golden_labels_task = golden_labels[task]
                print('value_scores:',value_scores)
                value_scores_task = value_scores[task][value].tolist()
                task_correct_predictions = 0
                task_total_predictions = len(full_answers_task)
                overall_total_predictions += len(full_answers_task)

                # Calculate task accuracy
                for this_ques, full_ans, golden, this_value_score in tqdm(
                    zip(ques_task, full_answers_task, golden_labels_task, value_scores_task), desc=f"Processing {task}"
                ):
                    prediction = self.extract_answer(full_ans)

                    score = self.scorer.exact_match_score(
                        golden, prediction
                    )
                    
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (task, this_ques, golden, prediction, score, this_value_score)
                    )

                task_accuracy = (
                    task_correct_predictions / task_total_predictions
                )
                print(
                    f"MMLU Task Accuracy (task={task}): {task_accuracy}"
                )
                scores_row.append((task, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall MMLU Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            # Columns: 'Task', 'Input', 'Prediction', 'Score'
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=["Task", "Question", "Golden", "Answer", "Correct", "Value Score"],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def load_benchmark_dataset(self, task: MMLUTask) -> List[Golden]:

        dataset = load_dataset(
            "lukaemon/mmlu", 
            task.value, 
            trust_remote_code=True, 
            cache_dir="./mmlu"
        )
        self.dataset = dataset

        # If dataset has not been previously loaded, construct
        # dataset of examples and save as instance var (to save time)
        if not self.shots_dataset:
            train_set = dataset["train"]
            shots_set = []
            for data in train_set:
                shots_set.append(data)
            self.shots_dataset = shots_set

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["test"]:
            input = MMLUTemplate.format_question(data, include_answer=False)
            golden = Golden(input=input, expected_output=data["target"])
            goldens.append(golden)
        return goldens

    def extract_answer(self, answer):
        answer_formats = [' {} ', '\n{}.', '({})', ' {}.', ' {}\n', '{}:', '\n{}\n', '{}.', '{},', '{}!', '"{}"', "'{}'"]
        for a_ans in [form.format('A') for form in answer_formats]:
            if (a_ans in answer) or (answer == 'A'):
                return 'A'
        for b_ans in [form.format('B') for form in answer_formats]:
            if b_ans in answer or (answer == 'B'):
                return 'B'
        for c_ans in [form.format('C') for form in answer_formats]:
            if c_ans in answer or (answer == 'C'):
                return 'C'
        for d_ans in [form.format('D') for form in answer_formats]:
            if d_ans in answer or (answer == 'D'):
                return 'D'  