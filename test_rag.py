from query_data import query_rag
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Arabic prompt template for evaluation
EVAL_PROMPT = """
الاستجابة المتوقعة: {expected_response}
الاستجابة الفعلية: {actual_response}
---
هل تطابق الاستجابة الفعلية الاستجابة المتوقعة؟
"""

# Load the Arabic BERT model
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-msa")
model = AutoModelForSequenceClassification.from_pretrained(
    "CAMeL-Lab/bert-base-arabic-camelbert-msa", num_labels=2
)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


def test_contractor_fee():
    print("اختبار أجر المتعاقد...")
    result = query_and_validate(
        question="كيف يدفع أجر المتعامل المتعاقد ؟",
        expected_response="السعر الإجمالي والجزافي، سعر الوحدة، سعر مختلط",
    )
    print(f"نتيجة اختبار أجر المتعاقد: {'نجح' if result else 'فشل'}\n")
    return result


def test_agricultural_investments():
    print("اختبار الاستثمارات الزراعية...")
    result = query_and_validate(
        question="ماهي المدة القانونية للعمل المرجعي في المستثمرات الفلاحية؟ (أجب بالرقم فقط)",
        expected_response="2000 ساعة",
    )
    print(f"نتيجة اختبار المستثمرات الفلاحية: {'نجح' if result else 'فشل'}\n")
    return result


def query_and_validate(question: str, expected_response: str):
    print(f"السؤال: {question}")
    print(f"الإجابة المتوقعة: {expected_response}")
    response_text = query_rag(question)
    print(f"الإجابة الفعلية: {response_text}")

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    # Use the Arabic BERT model for classification
    results = classifier(prompt)
    prediction = results[0]["label"]

    print("نص التقييم:")
    print(prompt)

    if prediction == "LABEL_1":  # Assuming LABEL_1 corresponds to "صحيح"
        print("\033[92m" + "نتيجة التقييم: صحيح" + "\033[0m")
        return True
    else:
        print("\033[91m" + "نتيجة التقييم: خطأ" + "\033[0m")
        return False


if __name__ == "__main__":
    print("بدء اختبارات RAG...\n")
    contractor_fee_result = test_contractor_fee()
    agricultural_investments_result = test_agricultural_investments()

    print("ملخص الاختبارات:")
    print(f"أجر المتعاقد: {'نجح' if contractor_fee_result else 'فشل'}")
    print(f"المستثمرات الفلاحية: {'نجح' if agricultural_investments_result else 'فشل'}")

    if contractor_fee_result and agricultural_investments_result:
        print("\nنجحت جميع الاختبارات!")
    else:
        print("\nفشلت بعض الاختبارات. يرجى مراجعة النتائج أعلاه.")
