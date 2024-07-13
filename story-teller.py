from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os

# Load environment variables from a .env file if present
load_dotenv()

def set_llm_env(api_key: str) -> None:
    """Set environment variables for LLM API key."""
    os.environ["OPENROUTER_API_KEY"] = api_key

def get_api_key() -> str:
    """Retrieve the API key from environment variables."""
    return os.getenv('OPENROUTER_API_KEY')

def get_task_result(task_output) -> str:
    """Extract the raw output from a task."""
    return task_output.raw_output

def create_openrouter_llm(model_name: str) -> ChatOpenAI:
    """Create and return a ChatOpenAI instance configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key not found in environment variables.")
    return ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

def main():
    api_key = get_api_key()

    if api_key is None:
        print("Error: API key is not set.")
        return

    set_llm_env(api_key)
    print("Environment variables set successfully.")

    print("## Welcome to the story generator.")
    scenario_input = input("Describe a scenario: ")

    try:
        story_model = "gryphe/mythomist-7b:free"
        critic_model = "nousresearch/nous-capybara-7b:free"

        story_agent = create_agent("Storyteller", "Create a compelling story based on the described scenario, embracing artistic freedom.", story_model)
        critic_agent = create_agent("Literary Critic", "Provide insightful and detailed feedback to enhance the quality of the story.", critic_model)

        tasks = create_tasks(scenario_input, story_agent, critic_agent)
        crew = Crew(
            agents=[story_agent, critic_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        print("Process completed.")
        handle_result(result)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def create_agent(role: str, goal: str, model_name: str) -> Agent:
    """Create and return an agent with the specified role, goal, and LLM model."""
    backstories = {
        "Storyteller": """
            You are a versatile and imaginative agent committed to artistic freedom and creativity.
            Your goal is to fully immerse in the scenario provided by the user, crafting a narrative that aligns with the given themes.
            Your stories are characterized by their originality, emotional depth, and adherence to the core elements of the user-defined scenario.
            You draw inspiration from a broad spectrum of literature and storytelling traditions, ensuring that each story is unique and true to its intended genre.
        """,
        "Literary Critic": """
            The critic agent, named "Aurelius," embodies the analytical prowess of history's greatest literary critics and scholars.
            Aurelius has spent its development in virtual literary salons, absorbing insights from classical and contemporary critiques.
            It evaluates narrative structure, character development, dialogue, style, coherence, and analytical rigor.
            Aurelius' mission is to guide writers on their creative journeys, helping them polish their narratives to shine with clarity, depth, and artistic merit.
        """
    }
    llm = create_openrouter_llm(model_name)
    return Agent(
        role=role,
        goal=goal,
        backstory=backstories[role],
        llm=llm
    )

def create_tasks(scenario_input: str, story_agent: Agent, critic_agent: Agent) -> list:
    """Create and return the list of tasks for the agents."""
    task1 = Task(
        description=f"Create a story based on the following scenario: {scenario_input}. Focus on the themes of fear, friendship, and survival.",
        agent=story_agent,
        expected_output="A well-crafted story that aligns with the given scenario."
    )

    task2 = Task(
        description="""
            Review the story and provide detailed feedback on the following elements:
            - Narrative structure
            - Character development
            - Dialogue
            - Style
            - Coherence
            - Any other relevant aspects

            The feedback should be constructive, actionable, and aimed at improving the overall quality of the prose.
        """,
        agent=critic_agent,
        expected_output="A comprehensive critique of the story, including specific suggestions and recommendations for revisions."
    )

    task3 = Task(
        description="Rewrite the original story, incorporating all the critical comments provided in the feedback.",
        agent=story_agent,
        expected_output="A revised version of the story that addresses all critical comments from the critique."
    )

    return [task1, task2, task3]

def handle_result(result):
    """Handle the result from the Crew kickoff process."""
    try:
        if hasattr(result, 'tasks'):
            print("Story Output:")
            print(get_task_result(result.tasks[0].output))
            print("Critique Output:")
            print(get_task_result(result.tasks[1].output))
            print("Revised Story Output:")
            print(get_task_result(result.tasks[2].output))
        else:
            print("Unexpected result format:")
            print(result)
    except Exception as e:
        print(f"An error occurred while handling the result: {str(e)}")

if __name__ == "__main__":
    main()
