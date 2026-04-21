# Very simple run file for testing purposes

from product_analysis import analyze_product

result = analyze_product(
    image_path="banana_r.jpg",
    checkpoint_path="FoodModel_1.pth",
    output_dir="analysis_outputs",
)

print(result)
