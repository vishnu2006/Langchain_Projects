import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = ""   # Replace with your key

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# ---------------- PROMPT TEMPLATE ---------------- #
prompt = ChatPromptTemplate.from_template("""
You are an expert mentor creating a structured **learning roadmap**.

Inputs:
- Learning topic: {topic}
- Duration: {duration} weeks
- Daily study time: {hours} hours/day
- Goal: {goal}
- Current knowledge level: {level}

Create a week-by-week roadmap that includes:
1. Key concepts to learn each week.
2. Suggested tutorials, docs, or YouTube searches (not links, just titles).
3. Mini-project ideas or practice tasks.
4. One tip per week for staying consistent.

Rules:
- Keep it **concise but actionable**.
- Organize it in a clear Week 1 → Week N format.
- Tailor based on user's level and goal.
""")

# ---------------- ROADMAP FUNCTION ---------------- #
def generate_learning_roadmap(topic, duration, hours, goal, level):
    chain = prompt | llm
    response = chain.invoke({
        "topic": topic,
        "duration": duration,
        "hours": hours,
        "goal": goal,
        "level": level
    })
    return response.content

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("🎓 AI Learning Roadmap Generator\n")

    topic = input("Enter your learning topic (e.g., Machine Learning, Web Dev, NLP): ")
    duration = int(input("Enter total duration (in weeks): "))
    hours = int(input("How many hours can you study per day? "))
    goal = input("What is your end goal? (e.g., Build a project / Get a job / Understand fundamentals): ")
    level = input("Your current knowledge level (Beginner / Intermediate / Advanced): ")

    print("\n🧠 Generating your personalized learning roadmap...\n")
    roadmap = generate_learning_roadmap(topic, duration, hours, goal, level)

    print("======= Your Learning Roadmap =======\n")
    print(roadmap)
    print("\n====================================")
