import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Set API key
os.environ["GOOGLE_API_KEY"] = ""   # replace with yours safely

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_template("""
You are a smart travel planner.
Create a **{days}-day travel itinerary** for:

- Destination: {destination}
- Budget level: {budget} (Low/Medium/High)
- Traveler type: {traveler_type} (Solo, Couple, Family, Friends)

Rules for output:
1. Keep it **short & practical** (only daily highlights).
2. Include **places to visit, food suggestions, and activities**.
3. Show it in a **Day 1, Day 2... format**.
4. Avoid unnecessary detail.
""")

# ---------------- ITINERARY FUNCTION ---------------- #
def generate_itinerary(destination, budget, days, traveler_type):
    chain = prompt | llm
    response = chain.invoke({
        "destination": destination,
        "budget": budget,
        "days": days,
        "traveler_type": traveler_type
    })
    return response.content

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("✈️ AI Travel Itinerary Maker\n")

    destination = input("Enter your destination: ")
    budget = input("Budget (Low/Medium/High): ")
    days = int(input("Number of days: "))
    traveler_type = input("Traveler type (Solo/Couple/Family/Friends): ")

    print("\n🧳 Generating your travel itinerary...\n")
    plan = generate_itinerary(destination, budget, days, traveler_type)

    print("======= Your Travel Itinerary =======\n")
    print(plan)
    print("\n====================================")
