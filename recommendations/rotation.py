import pandas as pd

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

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

# إنشاء مصفوفة الدوران: previous_crop -> recommended_next_crop
rotation = {}

# يمكن اختيار أول عنصر مناسب من المجموعة التالية
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

# اختبار المصفوفة
for k, v in rotation.items():
    print(f"{k} -> {v}")
