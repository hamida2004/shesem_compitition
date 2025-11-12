# recommendations/rotation.py

import pandas as pd

def build_rotation_matrix(csv_path="Crop_recommendation.csv"):
    """
    بناء مصفوفة الدوران للمحاصيل بناءً على نوعها.
    تُرجع dictionary: previous_crop -> list of recommended next crops
    """
    df = pd.read_csv(csv_path)

    # استبعاد القيم الفارغة والحصول على القيم الفريدة
    unique_labels = df['label'].dropna().unique().tolist()

    # تعريف مجموعات المحاصيل بناءً على النوع (باللغة العربية)
    legumes = [c for c in unique_labels if c in [
        'حمص', 'الفاصوليا الحمراء', 'البسلة الهندية', 'الفاصوليا العثة', 'المونج', 'الجرام الأسود', 'عدس', 'فول'
    ]]  # بقوليات (تثبيت النيتروجين)

    cereals_fibers = [c for c in unique_labels if c in [
        'أرز', 'ذرة', 'قمح', 'شعير', 'الجوت'
    ]]  # حبوب وألياف

    fruits = [c for c in unique_labels if c in [
        'موز', 'مانجو', 'عنب', 'بطيخ', 'شمام', 'تفاح', 'برتقال', 'بابايا', 'رمان'
    ]]

    others = [c for c in unique_labels if c in [
        'جوز الهند', 'قهوة', 'زيتون', 'طماطم', 'بطاطا'
    ]]  # أشجار/محاصيل أخرى

    # إنشاء مصفوفة الدوران
    rotation = {}

    # اختيار أول عنصر مناسب من المجموعة التالية
    for crop in cereals_fibers + fruits + others:
        rotation[crop] = legumes[:]  # تتبعها البقوليات
    for crop in legumes:
        rotation[crop] = cereals_fibers + fruits + others  # تتبعها المحاصيل الأخرى
    for crop in others:
        rotation[crop] = fruits + others  # الأشجار/محاصيل أخرى

    # إزالة التكرار الذاتي إذا وجد
    for crop in rotation:
        if crop in rotation[crop]:
            rotation[crop].remove(crop)

    return rotation


# اختبار سريع
if __name__ == "__main__":
    matrix = build_rotation_matrix()
    for k, v in matrix.items():
        print(f"{k} -> {v}")
