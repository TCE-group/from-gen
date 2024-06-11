from flask import Flask, request, jsonify
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import json

app = Flask(__name__)

# Define IBM Watson credentials and project ID
credentials = {
    'url': "https://us-south.ml.cloud.ibm.com",
    'apikey': "sfYw19CSGop6wNNNmqfPh9VDVe88eBbR55aP-zWphpnn"
}
project_id = "beaf6470-c5bc-4695-b204-29d09c8bf7fb"

# Define generation parameters
generation_params = {
    GenParams.MAX_NEW_TOKENS: 2500,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

# Define prompt templates
prompt_improve_input = PromptTemplate(
    input_variables=["text"],
    template="Improve the input based on the reference {text} to ensure accuracy."
)

prompt_combine_input = PromptTemplate(
    input_variables=["text"],
    template="Utilize the original input and corrections in {text} to improve the resulting medical form accuracy."
)

prompt_modify_input = PromptTemplate(
    input_variables=["text"],
    template="List corrections in {text} and hallucinated parts of information aside from section headings."
)

prompt_compare_results = PromptTemplate(
    input_variables=["text"],
    template="Compare the result against the input in {text}. Pinpoint 1) specific patient information in the input that is not in the output. 2) high-value information in the input. 3) list corrections and hallucinated parts of information. (Ignore semantically similar information)."
)

prompt_categorize_results = PromptTemplate(
    input_variables=["result"],
    template="A) Use 1) 2) 3) from {result} to update the result.  B) Categorize into demographics, Chief Complaint, HPI (history of patient illness), ROS (review of systems), PMHx (patient medical history), Social History, Family History, Physical Exam and medical tests, Medications and their dosages, treatment Plans and recommendations, corresponding billing and ICD codes. Avoid adding extra information not present in the text."
)

# Define model instances
meta_llama_model = ModelInference(
    model_id="meta-llama/llama-3-70b-instruct",
    credentials=credentials,
    params=generation_params,
    project_id=project_id
)
mix_model = ModelInference(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    credentials=credentials,
    params=generation_params,
    project_id=project_id
)
meta = WatsonxLLM(watsonx_model=meta_llama_model)

# Define LLM chains
context_chain = LLMChain(
    llm=meta,
    prompt=PromptTemplate(
        input_variables=["raw_text"],
        template="Carefully and honestly use {raw_text} as a conversation between a patient and a doctor."
    )
)

form_chain = LLMChain(
    llm=meta,
    prompt=prompt_categorize_results
)

compare_chain = LLMChain(
    llm=meta,
    prompt=prompt_compare_results
)

nonhallucinate_chain = LLMChain(
    llm=meta,
    prompt=prompt_modify_input
)

combine_chain = LLMChain(
    llm=meta,
    prompt=prompt_combine_input
)

finish_chain = LLMChain(
    llm=meta,
    prompt=prompt_improve_input
)


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello"

@app.route('/process_text', methods=['GET'])
def process_text():
    text = request.args.get('text')

    # Run the QA chain
    qa_chain = SimpleSequentialChain(chains=[context_chain, form_chain], verbose=True)
    result = qa_chain.invoke(text)

    # Assuming result is a dictionary containing your data
    result_json = json.dumps(str(result))

    # Find the index of "Demographics:"
    #index = result_json.find("Demographics:")
    #info = result_json[index:].strip()
    info = result_json.strip()

    # Print the result to the console
    print(info)

    # Prepare for finishing
    last_input_and_reference = f"Input: {text} ||| Reference: {info}"
    res = finish_chain.invoke(last_input_and_reference)


    combined = {**result, **res}

    # Convert the combined dictionary to JSON
    combined_json = jsonify(combined)
    

    return combined_json


if __name__ == '__main__':
    app.run(debug=True)
