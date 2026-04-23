import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- TEXT CLEAN FUNCTION ----------
def clean_text(text):
    text = str(text).lower()                 
    text = re.sub(r'[^a-z0-9 ]', ' ', text) 
    text = re.sub(r'\s+', ' ', text)         
    return text.strip()

# ---------- ALGORITHM MAP ----------
algo_map = {
    # ML / AI models
    "cnn": "cnn",
    "convolutional neural network": "cnn",
    "random forest": "random_forest",
    "decision tree": "decision_tree",
    "logistic regression": "logistic_regression",
    "svm": "svm",
    "support vector machine": "svm",
    "knn": "knn",
    "yolo": "yolo",
    "arima": "arima",
    "gan": "gan",
    "auto encoder": "autoencoder",
    "autoencoder": "autoencoder",
    "mobilenet": "mobilenet",
    "resnet": "resnet",
    "vgg": "vgg",
    "inception": "inception",
    "efficientnet": "efficientnet",
    "qsvm": "qsvm",
    "llm": "llm",
    "llama": "llm",
    "transformer": "transformer",

    # Rule-based / control
    "rule-based": "rule_based",
    "threshold-based": "threshold_based",
    "event-driven": "event_driven",
    "time-based": "time_based",

    # Text similarity algorithms (YOU chose to keep)
    "tf-idf": "tfidf",
    "tf idf": "tfidf",
    "cosine": "cosine_similarity",
    "cosine similarity": "cosine_similarity" ,
    "jaccard similarity": "jaccard_similarity" ,
}

# ---------- EXTRACT ALGORITHMS ----------
def extract_algorithms_strict(text):
    text = str(text).lower()
    found_algos = set()
    
    for key, standard_name in algo_map.items():
        if key in text:
            found_algos.add(standard_name)
    
    if len(found_algos) == 0:
        return "none"   # <-- IMPORTANT
    
    return " ".join(sorted(found_algos))

# ---------- DOMAIN MAP ----------
domain_map = {
    # AI / ML domains
    "machine learning": "ml",
    "ml": "ml",
    "deep learning": "dl",
    "dl": "dl",
    "artificial intelligence": "ai",
    "ai": "ai",
    "computer vision": "computer_vision",
    "image processing": "computer_vision",
    "nlp": "nlp",
    "natural language": "nlp",
    "llm": "nlp",
    "generative ai": "ai",

    # IoT / Embedded
    "iot": "iot",
    "internet of things": "iot",
    "embedded": "embedded_systems",

    # Other domains
    "data science": "data_science",
    "blockchain": "blockchain",
    "cyber": "cyber_security",
    "web": "web",
    "agriculture": "agriculture",
    "healthcare": "healthcare",
    "quantum": "quantum_computing"
}

# ---------- EXTRACT DOMAIN ----------
def extract_domain_strict(text):
    text = str(text).lower()
    found = set()
    
    for key, value in domain_map.items():
        if key in text:
            found.add(value)
    
    if len(found) == 0:
        return "none"
    
    return " ".join(sorted(found))

# ---------- JACCARD ----------
def jaccard_similarity(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    
    if "none" in set_a and "none" in set_b:
        return 1.0
    if len(set_a.union(set_b)) == 0:
        return 0
    
    return len(set_a & set_b) / len(set_a | set_b)


#-----------Recommendation logic-------
def recommendation_logic(title_sim, algo_sim, domain_sim):

    # Thresholds (tuned for student projects)
    TITLE_HIGH = 0.6
    TITLE_MED = 0.3
    ALGO_HIGH = 0.5
    DOMAIN_HIGH = 0.6

    #Exact Duplicate Project
    if title_sim >= TITLE_HIGH and algo_sim >= ALGO_HIGH and domain_sim >= DOMAIN_HIGH:
        return "CHANGE PROJECT IDEA (Exact Duplicate)"

    #Same topic, different method
    if title_sim >= TITLE_HIGH and domain_sim >= DOMAIN_HIGH and algo_sim < ALGO_HIGH:
        return "MODIFY APPROACH / METHODOLOGY"

    #Different topic but same methodology reused
    if title_sim < TITLE_HIGH and algo_sim >= ALGO_HIGH and domain_sim >= DOMAIN_HIGH:
        return "INNOVATION WARNING: Methodology Reused"

    #Similar title only
    if title_sim >= TITLE_HIGH and domain_sim < DOMAIN_HIGH:
        return "CHANGE TITLE"

    #Everything low → unique
    return "GOOD TO GO (Unique Project)"




# ---------- LOAD DATA ----------
df = pd.read_csv("project similarity dataset.csv")
df = df.replace("", "None")
df = df.fillna("None")

# Clean dataset columns
df["clean_title"] = df["TITLE OF THE PROJECT"].apply(clean_text)
df["clean_algorithms_final"] = df["ALGORITHMS USED IN THE PROJECT"].apply(extract_algorithms_strict)
df["clean_domain_final"] = df["DOMAIN USED IN THE PROJECT"].apply(extract_domain_strict)


# ---------- STREAMLIT UI ----------
st.title("Project Similarity Detection System")

user_title = st.text_input("Enter the Project Title")
user_algorithms = st.text_input("Enter the Algorithms Used")
user_domain = st.text_input("Enter the Techniques Used")

# ---------------- CHECK SIMILARITY ----------------
if st.button("Check Similarity"):
    
    # Clean user input
    user_title_clean = clean_text(user_title)
    user_algo_clean = extract_algorithms_strict(user_algorithms)
    user_domain_clean = extract_domain_strict(user_domain)

    # TITLE SIMILARITY
    all_titles = df["clean_title"].tolist() + [user_title_clean]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_titles)
    df["title_similarity"] = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # ALGO SIMILARITY
    df["algo_similarity"] = df["clean_algorithms_final"].apply(
        lambda x: jaccard_similarity(user_algo_clean, x)
    )

    # DOMAIN SIMILARITY
    df["domain_similarity"] = df["clean_domain_final"].apply(
        lambda x: jaccard_similarity(user_domain_clean, x)
    )

    # RANK SCORE
    df["rank_score"] = (0.5*df["title_similarity"] +
                        0.3*df["algo_similarity"] +
                        0.2*df["domain_similarity"])

    # TOP 5
    top5 = df.sort_values("rank_score", ascending=False).head(5)

    # SAVE TOP5 IN MEMORY
    st.session_state["top5"] = top5

    st.subheader("Top 5 Similar Projects")
    st.dataframe(top5[["TITLE OF THE PROJECT","rank_score"]])
    
# ---------------- SHOW DETAILS ----------------
if "top5" in st.session_state:
    
    project_titles = st.session_state["top5"]["TITLE OF THE PROJECT"].tolist()
    selected_project = st.radio("Select a project to view details:", project_titles)

    if st.button("Show Details"):
        row = st.session_state["top5"][st.session_state["top5"]["TITLE OF THE PROJECT"] == selected_project].iloc[0]

        rec = recommendation_logic(row["title_similarity"], row["algo_similarity"], row["domain_similarity"])

        st.subheader("Project Details")
        st.write("Title:", row["TITLE OF THE PROJECT"])
        st.write("Domain:", row["DOMAIN USED IN THE PROJECT"])
        st.write("Algorithms:", row["ALGORITHMS USED IN THE PROJECT"])
        st.write("Dataset:", row["DATASET USED(Name & Link)"])

        st.subheader("Similarity Scores")
        st.write("Title Similarity:", round(row["title_similarity"], 3))
        st.write("Algorithm Similarity:", round(row["algo_similarity"], 3))
        st.write("Domain Similarity:", round(row["domain_similarity"], 3))

        st.subheader("Recommendation")
        st.success(rec)

