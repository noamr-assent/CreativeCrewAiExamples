import datetime
import os
import json
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def set_llm_env(api_key: str) -> None:
    """Set environment variables for LLM API key."""
    os.environ["OPENROUTER_API_KEY"] = api_key

def get_api_key() -> str:
    """Retrieve the API key from environment variables."""
    return os.getenv('OPENROUTER_API_KEY')

def get_task_result(task_output):
    """Extract the raw output from a task, handling both string and object outputs."""
    if isinstance(task_output, str):
        return task_output
    elif hasattr(task_output, 'raw_output'):
        return task_output.raw_output
    else:
        return str(task_output)  # Fall back to string representation if neither condition is met

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

def create_agent(role: str, goal: str, model_name: str) -> Agent:
    """Create and return an agent with the specified role, goal, and LLM model."""
    backstories = {
        "Card Creator": """
            You are an expert in creating rich, detailed character profiles for role-playing games and interactive fiction.
            You understand the importance of using placeholders for dynamic character interactions.
        """,
        "Personality Critic": """
            You are a master of character development, skilled at identifying and enhancing personality traits to create compelling characters.
        """,
        "Scenario Critic": """
            You are an experienced storyteller, adept at crafting intriguing scenarios that bring characters to life.
        """,
        "Message Critic": """
            You are a dialogue expert, skilled at crafting authentic and captivating character speech with proper formatting and placeholder usage.
        """
    }
    llm = create_openrouter_llm(model_name)
    return Agent(
        role=role,
        goal=goal,
        backstory=backstories[role],
        llm=llm
    )

# Function to create character card JSON
def create_character_card(name, description, personality, scenario, first_mes, mes_example, creatorcomment, chat, talkativeness, fav, tags, create_date):
    return {
        "name": name,
        "description": description,
        "personality": personality,
        "scenario": scenario,
        "first_mes": first_mes,
        "mes_example": mes_example,
        "creatorcomment": creatorcomment,
        "avatar": "none",
        "chat": chat,
        "talkativeness": talkativeness,
        "fav": fav,
        "tags": tags,
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": description,
            "personality": personality,
            "scenario": scenario,
            "first_mes": first_mes,
            "mes_example": mes_example,
            "creator_notes": creatorcomment,
            "system_prompt": "",
            "post_history_instructions": "",
            "tags": tags,
            "creator": "",
            "character_version": "",
            "alternate_greetings": [],
            "extensions": {
                "talkativeness": talkativeness,
                "fav": fav,
                "world": "",
                "depth_prompt": {
                    "prompt": "",
                    "depth": 1,
                    "role": "system"
                }
            }
        },
        "create_date": create_date
    }

def main():
    print("""
    Welcome to the TavernAI character card creator. 
    Please provide a free-form description of the character you want to create.
    Include any details you think are important, such as name, personality, background, etc.
    The AI will use this information to create a complete character card.
    """)

    api_key = get_api_key()
    if api_key is None:
        print("Error: API key is not set.")
        return

    set_llm_env(api_key)
    print("Environment variables set successfully.")

    user_input = input("Enter your character description: ").strip()

    # Set default values
    chat = f"Character - {datetime.datetime.now().strftime('%Y-%m-%d @%Hh %Mm %Ss %fms')}"
    talkativeness = 0.5
    fav = False
    tags = []
    create_date = datetime.datetime.now().strftime('%Y-%m-%d @%Hh %Mm %Ss %fms')

    # Define agents
    card_creator_agent = create_agent("Card Creator", "Create a detailed and engaging character card based on the user's free-form input.", "google/gemma-2-9b-it:free")
    personality_critic_agent = create_agent("Personality Critic", "Ensure the character has a well-rounded and interesting personality.", "meta-llama/llama-3-8b-instruct:free")
    scenario_critic_agent = create_agent("Scenario Critic", "Ensure the character's scenario is engaging, interesting, and consistent with their personality.", "meta-llama/llama-3-8b-instruct:free")
    message_critic_agent = create_agent("Message Critic", "Ensure the opening message is engaging and properly formatted.", "meta-llama/llama-3-8b-instruct:free")

    # Define tasks
    tasks = [
        Task(
            description=f"Create a character card based on the following user input: {user_input}. Include name, description, personality, scenario, first message, example message, and creator comment. Use {{{{char}}}} to refer to the character being created and {{{{user}}}} for the user interacting with the character.",
            agent=card_creator_agent,
            expected_output="A JSON string containing the character card details."
        ),
        Task(
            description="Review and enhance the personality of the character. Ensure {{char}} is used consistently to refer to the character.",
            agent=personality_critic_agent,
            expected_output="A string containing feedback and suggestions for the character's personality."
        ),
        Task(
            description="Review and improve the scenario for the character. Verify that {{char}} and {{user}} are used appropriately in the scenario.",
            agent=scenario_critic_agent,
            expected_output="A string containing feedback and suggestions for the character's scenario."
        ),
        Task(
            description="Review and refine the first message and example message for the character. Ensure proper formatting with *italics* for narration and \"quotes\" for dialogue. Confirm that {{char}} and {{user}} are used correctly in the messages.",
            agent=message_critic_agent,
            expected_output="A string containing feedback and suggestions for the character's messages."
        )
    ]

        # Set up the crew
    crew = Crew(
        agents=[card_creator_agent, personality_critic_agent, scenario_critic_agent, message_critic_agent],
        tasks=tasks,
        verbose=True
    )

    try:
        # Kick off the process
        result = crew.kickoff()
        print("Process completed.")
        
        # Extract the final character card from the result
        if isinstance(result, list) and len(result) > 0:
            character_card_json = get_task_result(result[0])
        else:
            character_card_json = get_task_result(result)
        
        # Parse the JSON string into a Python dictionary
        try:
            character_card = json.loads(character_card_json)
        except json.JSONDecodeError:
            print("Error: Unable to parse the character card JSON. Using raw output.")
            character_card = {"name": "Unnamed Character", "description": character_card_json}

        # Create the final character card
        final_card = create_character_card(
            name=character_card.get('name', 'Unnamed Character'),
            description=character_card.get('description', ''),
            personality=character_card.get('personality', ''),
            scenario=character_card.get('scenario', ''),
            first_mes=character_card.get('first_message', ''),
            mes_example=character_card.get('example_message', ''),
            creatorcomment=character_card.get('creator_comment', ''),
            chat=chat,
            talkativeness=character_card.get('talkativeness', talkativeness),
            fav=character_card.get('fav', fav),
            tags=character_card.get('tags', tags),
            create_date=create_date
        )

        # Print the final character card
        print("\nFinal Character Card:")
        print(json.dumps(final_card, indent=2))

        # Save the character card to a file
        file_name = f"{final_card['name'].lower().replace(' ', '_')}_character_card.json"
        with open(file_name, 'w') as f:
            json.dump(final_card, f, indent=2)
        print(f"\nCharacter card saved to {file_name}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()