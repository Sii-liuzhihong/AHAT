import argparse
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ahat.planning.decompose import TaskDecomposer
from ahat.planning.solve import SolveCoordinator
from ahat.configs.constants import AHAT_MODEL_DEFAULT_PATH, AHAT_DATASET_DEFAULT_FILE

class AHATPipeline:
    """
    Orchestrates the complete AHAT planning pipeline.
    
    Workflow:
    1. Load input data (instructions and scene graphs)
    2. Decompose tasks into subgoals using LLM (TaskDecomposer)
    3. Solve PDDL plans for each decomposed subgoal (SolveCoordinator)
    4. Aggregate and save results
    """

    def __init__(self, data_path: Path, domain_file_path: Path, output_dir: Path,
                 decomposer_mode: str = "local", decomposer_model_name: Optional[str] = None):
        """
        Initialize the pipeline with configuration paths and task decomposer.
        
        Args:
            data_path: Path to the input data JSON file.
            domain_file_path: Path to the PDDL domain file.
            output_dir: Path to the output directory.
            decomposer_mode: Mode for TaskDecomposer ("api" or "local").
            decomposer_model_name: Model name/path for TaskDecomposer.
        """
        self.data_path = Path(data_path)
        self.domain_file_path = Path(domain_file_path)
        self.output_dir = Path(output_dir)
        self.pddl_solver_output_dir = self.output_dir / "pddl_solver_output"
        self.output_path = self.output_dir / "pipeline_results.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Initialized TaskDecomposer with mode={decomposer_mode} and model={decomposer_model_name}")
        
        # Initialize TaskDecomposer as an instance attribute
        self.task_decomposer = TaskDecomposer(
            mode=decomposer_mode,
            model_name=decomposer_model_name
        )

    def run(self, max_items: int = None) -> list:
        """
        Execute the complete pipeline.
        
        Args:
            max_items: Maximum number of items to process (None = all).
            
        Returns:
            List of result dicts with plan information and metadata.
        """
        print(f"Loading data from {self.data_path}...")
        try:
            with self.data_path.open("r", encoding="utf-8") as f:
                # Support both JSON array and JSONL formats
                content = f.read().strip()
                if content.startswith("["):
                    # JSON array format
                    data = json.loads(content)
                else:
                    # JSONL format (one JSON object per line)
                    data = [json.loads(line) for line in content.split("\n") if line.strip()]
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
        
        # Limit processing if specified
        if max_items is not None:
            data = data[:max_items]
        
        total_items = len(data)
        
        results = []
        for idx, item in enumerate(data):
            result = self._process_item(item, idx, total_items)
            results.append(result)
        
        self._save_results(results)
        return results

    def _process_item(self, item: dict, idx: int, total: int) -> dict:
        """
        Process a single data item through decomposition and solving.
        
        Args:
            item: Single data item with instruction and scene_graph.
            idx: Current item index.
            total: Total number of items.
            
        Returns:
            Result dict with decomposition and planning information.
        """
        instruction = item.get("instruction", "")
        scene_graph = item.get("scene_graph", {})
        task_id = item.get("id", f"task_{idx}")
        current_output_dir = self.pddl_solver_output_dir / task_id

        item_result = {
            "instruction": instruction,
            "scene_graph": scene_graph,
            "subgoal_reply": None,
            "plan_result": None
        }
        
        print(f"--- Processing item {idx + 1}/{total} ---")
        print(f"Instruction: {instruction}")
        
        # 1. Decompose task to get subgoals
        print("Calling LLM to generate subgoals...")
        try:
            subgoal_reply = self.task_decomposer.generate(instruction, scene_graph)
            item_result["subgoal_reply"] = subgoal_reply
            print(f"LLM Response:\n{subgoal_reply}\n")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            item_result["subgoal_reply"] = f"Error: {e}"
            return item_result
        
        # 2. Solve plan using PDDL solver based on LLM output
        print("Solving PDDL for plan...")
        try:
            plan_result = SolveCoordinator.solve(
                subgoal_reply, scene_graph, str(self.domain_file_path), str(current_output_dir)
            )
            item_result["plan_result"] = plan_result
            
            if plan_result.get("success"):
                print("Plan solved successfully!")
                print("Plan actions:")
                for action in plan_result.get("plan", []):
                    print(f"  {action}")
            else:
                print("Failed to solve plan.")
                print(f"Parsable: {plan_result.get('parsable')}")
                print(f"Subgoal List: {plan_result.get('subgoal_list')}")
        except Exception as e:
            item_result["plan_result"] = {"error": str(e)}
            print(f"Error solving plan: {e}")
        
        print("=" * 80)
        return item_result

    def _save_results(self, results: list) -> None:
        """
        Save pipeline results to output JSON file.
        
        Args:
            results: List of result dicts from pipeline execution.
        """
        print(f"Saving all results to {self.output_path}...")
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Done!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AHAT planning pipeline.")
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help="Decomposer mode: local or api (default: local)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to input dataset file (default: AHAT_DATASET_DEFAULT_FILE)",
    )
    parser.add_argument(
        "--domain-file-path",
        default=None,
        help="Path to the PDDL domain file (default: examples/ahat_domain.pddl)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for pipeline outputs (default: outputs/pipeline_results)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name/path",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Limit number of items to process",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the pipeline.
    Loads environment variables and executes the pipeline with default paths.
    """
    load_dotenv()

    args = _parse_args()

    # Calculate the path dynamically based on this file's location
    project_root = Path(__file__).resolve().parents[3]

    data_path = Path(args.data_path) if args.data_path else (project_root / AHAT_DATASET_DEFAULT_FILE)
    domain_file_path = (
        Path(args.domain_file_path) if args.domain_file_path else (project_root / "examples" / "ahat_domain.pddl")
    )
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / "pipeline_results")

    pipeline = AHATPipeline(
        data_path=data_path,
        domain_file_path=domain_file_path,
        output_dir=output_dir,
        decomposer_mode=args.mode,
        decomposer_model_name=args.model_name or AHAT_MODEL_DEFAULT_PATH,
    )
    pipeline.run(max_items=args.max_items)


if __name__ == "__main__":
    main()
