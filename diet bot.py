import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBnFn9dnDyLBEH2kO6ObDWne0-QD5MQsbA"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# ---------------- PROMPT ---------------- #
prompt = ChatPromptTemplate.from_template("""
You are a professional dietitian.
Create a **7-day concise diet plan** for:
- Gender: {gender}
- Age: {age}
- Weight: {weight} kg
- Height: {height} cm
- Diet preference: {diet_type}

Rules for output:
1. Keep it **short and practical**.
2. Only list **Breakfast, Lunch, Dinner, Snack** with small portion suggestions.
3. Mention daily calorie target.
4. Avoid too much detail.
""")

# ---------------- PLAN FUNCTION ---------------- #
def generate_diet_plan(gender, age, weight, height, diet_type):
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({
        "gender": gender,
        "age": age,
        "weight": weight,
        "height": height,
        "diet_type": diet_type
    })
    return response["text"]


if __name__ == "__main__":
    print("🍏 AI Diet Planner (Concise Version)\n")

    gender = input("Enter your gender (Male/Female): ")
    age = int(input("Enter your age: "))
    weight = float(input("Enter your weight (kg): "))
    height = int(input("Enter your height (cm): "))
    diet_type = input("Diet type (Veg/Non-Veg/Eggetarian): ")

    print("\n🧠 Generating your concise diet plan...\n")
    plan = generate_diet_plan(gender, age, weight, height, diet_type)

    print("======= Your 7-Day Diet Plan =======\n")
    print(plan)
    print("\n====================================")
