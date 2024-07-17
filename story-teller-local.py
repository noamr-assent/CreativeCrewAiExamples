import os
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

def main():
    print("## Welcome to the story generator.")
    scenario_input = input("Describe a scenario: ")

    try:
        # Create Ollama LLM for the story agent
        story_llm = Ollama(model="tiamat")

        story_agent = Agent(
            role="Storyteller",
            goal="Create a compelling story based on the described scenario, embracing artistic freedom.",
            backstory="""
                You are a versatile and imaginative agent committed to artistic freedom and creativity.
                Your goal is to fully immerse in the scenario provided by the user, crafting a narrative that aligns with the given themes.
                Your stories are characterized by their originality, emotional depth, and adherence to the core elements of the user-defined scenario.
                You draw inspiration from a broad spectrum of literature and storytelling traditions, ensuring that each story is unique and true to its intended genre.
            """,
            llm=story_llm
        )

        # Create Ollama LLM for the critic agent
        critic_llm = Ollama(model="gemma2")

        critic_agent = Agent(
            role="Literary Critic",
            goal="Provide insightful and detailed feedback to enhance the quality of the story.",
            backstory=""" 
                The critic agent, named "Aurelius," embodies the analytical prowess of history's greatest literary critics and scholars.
                Aurelius has spent its development in virtual literary salons, absorbing insights from classical and contemporary critiques.
                It evaluates narrative structure, character development, dialogue, style, coherence, and analytical rigor.
                Aurelius' mission is to guide writers on their creative journeys, helping them polish their narratives to shine with clarity, depth, and artistic merit.
            """,
            llm=critic_llm
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
            expected_output="A comprehensive critique of the story, including specific suggestions for improvement."
        )

        task3 = Task(
            description="Rewrite the original story, incorporating all the critical comments provided in the feedback.",
            agent=story_agent,
            expected_output="A revised version of the story that addresses the critique and improves upon the original."
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
        print(result[0])
        print("Critique Output:")
        print(result[1])
        print("Revised Story Output:")
        print(result[2])

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()