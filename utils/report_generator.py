from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# Load embeddings and FAISS vector store
# --------------------------------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = FAISS.load_local(
    "rag/vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# --------------------------------------------------
# Initialize local LLM (Ollama)
# --------------------------------------------------
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.3  # lower temperature for safer medical responses
)

output_parser = StrOutputParser()


# --------------------------------------------------
# Prompt templates
# --------------------------------------------------
patient_prompt = ChatPromptTemplate.from_template(
    """
    You are a compassionate medical assistant.

    Explain the diagnosed skin condition in simple, non-technical language
    so that a patient can easily understand it.

    Disease: {disease}
    Prediction confidence: {confidence}
    Lesion details: {lesion_info}

    Medical reference context:
    {context}

    Instructions:
    - Avoid medical jargon
    - Reassure the patient
    - Explain what the condition is
    - Mention general care advice
    - Clearly state that this is NOT a final diagnosis
    """
)

doctor_prompt = ChatPromptTemplate.from_template(
    """
    You are an experienced dermatologist preparing a clinical support report.

    Disease: {disease}
    Model confidence: {confidence}
    Lesion metrics: {lesion_info}

    Reference medical literature:
    {context}

    Instructions:
    - Use appropriate clinical terminology
    - Explain diagnostic reasoning
    - Mention possible differential diagnoses if relevant
    - Suggest next clinical steps (biopsy, dermoscopy, follow-up)
    - Clearly state that this is an AI-assisted opinion
    """
)


# --------------------------------------------------
# Report generation
# --------------------------------------------------
def generate_reports(disease, confidence, lesion_info):
    """
    Generate patient-friendly and doctor-focused reports using RAG.

    Parameters:
    disease      : Predicted disease name
    confidence   : Model confidence score
    lesion_info  : Information derived from segmentation

    Returns:
    patient_report : Simplified explanation for patients
    doctor_report  : Clinical explanation for doctors
    """

    # Retrieve relevant medical documents
    documents = vector_store.similarity_search(disease, k=3)
    context = "\n\n".join(doc.page_content for doc in documents)

    # Patient report
    patient_chain = patient_prompt | llm | output_parser
    patient_report = patient_chain.invoke({
        "disease": disease,
        "confidence": f"{confidence * 100:.2f}%",
        "lesion_info": lesion_info,
        "context": context
    })

    # Doctor report
    doctor_chain = doctor_prompt | llm | output_parser
    doctor_report = doctor_chain.invoke({
        "disease": disease,
        "confidence": f"{confidence * 100:.2f}%",
        "lesion_info": lesion_info,
        "context": context
    })

    return patient_report, doctor_report
