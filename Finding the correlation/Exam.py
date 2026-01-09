try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✓ Libraries loaded successfully")
except ImportError:
    print("Installing required libraries...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas", "matplotlib"])
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


x_values = [-9.00, -7.00, -5.00, -3.00, -1.00, 1.00, 3.00, 5.00, 7.00, 9.20]
y_values = [-8.30, -6.50, -4.70, -2.90, -0.80, 0.90, 3.10, 4.80, 6.70, 7.90]

print("\n" + "="*60)
print("PEARSON CORRELATION ANALYSIS")
print("="*60)


df = pd.DataFrame({
    'x_coordinate': x_values,
    'y_coordinate': y_values
})


df.to_csv('coordinates.csv', index=False)
print(f"\n✅ Data saved to 'coordinates.csv'")
print("First 5 rows:")
print(df.head())


correlation = np.corrcoef(x_values, y_values)[0, 1]


n = len(x_values)
mean_x = np.mean(x_values)
mean_y = np.mean(y_values)


covariance = np.sum((x_values - mean_x) * (y_values - mean_y))
std_x = np.std(x_values, ddof=1)  
std_y = np.std(y_values, ddof=1)
correlation_manual = covariance / ((n - 1) * std_x * std_y)

print(f"\n📊 STATISTICAL ANALYSIS")
print(f"Number of data points: {n}")
print(f"Mean of X: {mean_x:.4f}")
print(f"Mean of Y: {mean_y:.4f}")
print(f"Standard deviation of X: {std_x:.4f}")
print(f"Standard deviation of Y: {std_y:.4f}")

print(f"\n🔢 PEARSON CORRELATION COEFFICIENT")
print(f"Using numpy.corrcoef(): {correlation:.6f}")
print(f"Manually calculated:    {correlation_manual:.6f}")


def interpret_correlation(r_value):
    """Returns interpretation of correlation coefficient"""
    abs_r = abs(r_value)
    
    if abs_r >= 0.9:
        strength = "Very strong"
    elif abs_r >= 0.7:
        strength = "Strong"
    elif abs_r >= 0.5:
        strength = "Moderate"
    elif abs_r >= 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if r_value > 0 else "negative"
    return f"{strength} {direction} correlation"

interpretation = interpret_correlation(correlation)
print(f"\n📈 INTERPRETATION: {interpretation}")


m, b = np.polyfit(x_values, y_values, 1) 
print(f"\n📐 LINEAR REGRESSION LINE")
print(f"Equation: y = {m:.4f}x + {b:.4f}")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")


plt.figure(figsize=(12, 8))


plt.scatter(x_values, y_values, 
           color='blue', 
           s=200,           # წერტილების ზომა
           alpha=0.7,       # გამჭვირვალობა
           edgecolors='black',
           linewidth=2,
           label='Data points',
           zorder=3)


x_line = np.linspace(min(x_values) - 1, max(x_values) + 1, 100)
y_line = m * x_line + b
plt.plot(x_line, y_line, 
        color='red', 
        linestyle='--', 
        linewidth=3,
        label=f'Regression line: y={m:.2f}x+{b:.2f}',
        zorder=2)


for i, (x, y) in enumerate(zip(x_values, y_values)):
    plt.text(x, y + 0.3, f'({x},{y})', 
             fontsize=9, 
             ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))


plt.title('Pearson Correlation Analysis of Data Points', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('X Coordinate', fontsize=14, labelpad=10)
plt.ylabel('Y Coordinate', fontsize=14, labelpad=10)


plt.grid(True, alpha=0.3, linestyle='-', which='both')


info_text = f"""Pearson r = {correlation:.4f}
{interpretation}
n = {n} points
y = {m:.3f}x + {b:.3f}"""

plt.text(0.02, 0.98, info_text,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=10))


plt.legend(loc='lower right', fontsize=12)


plt.xlim(min(x_values) - 2, max(x_values) + 2)
plt.ylim(min(y_values) - 2, max(y_values) + 2)


plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
print(f"\n🖼️  Visualization saved as 'correlation_plot.png'")


plt.show()

print("\n" + "="*60)
print("ADDITIONAL CALCULATIONS")
print("="*60)


r_squared = correlation ** 2
print(f"R-squared (Coefficient of determination): {r_squared:.4f}")
print(f"This means {r_squared*100:.1f}% of Y's variation is explained by X")


sample_x = 0
predicted_y = m * sample_x + b
print(f"\n📝 PREDICTION EXAMPLE:")
print(f"For x = {sample_x}, predicted y = {predicted_y:.2f}")

predicted_values = m * np.array(x_values) + b
errors = np.array(y_values) - predicted_values
rmse = np.sqrt(np.mean(errors**2))
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE")
print("="*60)
print("Files created:")
print("1. coordinates.csv - contains all data points")
print("2. correlation_plot.png - visualization graph")
print(f"\nFinal Pearson correlation coefficient: {correlation:.6f}")