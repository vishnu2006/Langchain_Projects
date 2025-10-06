import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBnFn9dnDyLBEH2kO6ObDWne0-QD5MQsbA"   # replace with your key safely

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_template("""
You are a professional study planner.
Create a **day-wise study timetable** for a student preparing for exams.

Inputs:
- Subjects: {subjects}
- Priority subjects: {priority}
- Days till exam: {days}
- Available study window: {study_window}
- Daily study hours: {hours}

Rules:
1. Allocate more time to **priority subjects** while still covering all.
2. Place study sessions inside the **given time window**.
3. Divide the time into sessions with **short breaks**.
4. Keep the plan **clear, concise, and practical**.
5. Show output in **Day 1, Day 2... format with actual times**.
""")

# ---------------- TIMETABLE FUNCTION ---------------- #
def generate_timetable(subjects, priority, days, hours, study_window):
    chain = prompt | llm
    response = chain.invoke({
        "subjects": subjects,
        "priority": priority,
        "days": days,
        "hours": hours,
        "study_window": study_window
    })
    return response.content

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("📚 AI Study Timetable Generator (Smart Version)\n")

    subjects = input("Enter subjects (comma separated): ")
    priority = input("Enter priority subjects (comma separated, most important first): ")
    days = int(input("Enter number of days till exam: "))
    hours = int(input("Enter daily study hours: "))
    study_window = input("Enter available study window (e.g., 4 PM - 12 AM): ")

    print("\n🧠 Generating your personalized study timetable...\n")
    plan = generate_timetable(subjects, priority, days, hours, study_window)

    print("======= Your Study Timetable =======\n")
    print(plan)
    print("\n====================================")