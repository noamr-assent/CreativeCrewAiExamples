import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

def set_llm_env(api_key, model_name):
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["MODEL_NAME"] = model_name

def main():
    api_key = get_api_key()
    model_name = get_model_name()
    if api_key is None:
        print("Error: API key is not set.")
        return
    set_llm_env(api_key, model_name)
    print("Environment variables set successfully.")

def get_api_key():
    return os.getenv('OPENROUTER_API_KEY')

def get_model_name():
    return "meta-llama/llama-3-8b-instruct:free"

def get_task_result(task_output):
    # Assuming task_output is an object with a 'raw_output' attribute
    return task_output.raw_output

def create_openrouter_llm():
    return ChatOpenAI(
        model_name="meta-llama/llama-3-8b-instruct:free",  # or any other model you prefer
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1"        
    )

if __name__ == "__main__":
    main()

    print("## Welcome to the story generator.")
    scenario_input = input("Describe a scenario: ")

    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key not found in environment variables.")
        
        set_llm_env(openrouter_api_key, "meta-llama/llama-3-8b-instruct:free")
        story_agent = Agent(
            role="Storyteller",
            goal="Create a compelling story based on the described scenario, embracing artistic freedom.",
            backstory="""
                You are a versatile and imaginative agent committed to artistic freedom and creativity.
                Your goal is to fully immerse in the scenario provided by the user, crafting a narrative that aligns with the given themes.
                Your stories are characterized by their originality, emotional depth, and adherence to the core elements of the user-defined scenario.
                You draw inspiration from a broad spectrum of literature and storytelling traditions, ensuring that each story is unique and true to its intended genre.
            """,
            llm=create_openrouter_llm()
        )

        critic_agent = Agent(
            role="Literary Critic",
            goal="Provide insightful and detailed feedback to enhance the quality of the story.",
            backstory=""" 
                The critic agent, named "Aurelius," embodies the analytical prowess of history's greatest literary critics and scholars.
                Aurelius has spent its development in virtual literary salons, absorbing insights from classical and contemporary critiques.
                It evaluates narrative structure, character development, dialogue, style, coherence, and analytical rigor.
                Aurelius' mission is to guide writers on their creative journeys, helping them polish their narratives to shine with clarity, depth, and artistic merit.
            """,
            llm=create_openrouter_llm()
        )

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

        crew = Crew(
            agents=[story_agent, critic_agent],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        print("Process completed.")
        print("Story Output:")
        print(get_task_result(result.tasks[0].output))
        print("Critique Output:")
        print(get_task_result(result.tasks[1].output))
        print("Revised Story Output:")
        print(get_task_result(result.tasks[2].output))

    except Exception as e:
        print(f"An error occurred: {str(e)}")