import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Wedge, Circle

sns.set_palette('pastel')
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['text.color'] = '#333'

np.random.seed(0)
x = np.linspace(2, 12, 15)
y = 2.2 * x + np.random.normal(0, 3, 15)

important_features = {
    'Age':      (6.5,  1),
    'Sleep':    (6.0, -1),
    'Alcohol':  (5.3, -1),
    'Exercise': (4.9, -1),
    'Weight':       (4.2,  1),
    'Diet':     (4.0,  1),
    'Smoking':      (3.0, -1),
    'Marital status':      (2.1, -1),
    'Hours indoors':      (6.0, -1),
    'Income':       (1.0,  1)
}

shap_cols = ['Age', 'Sleep', 'Alcohol', 'Exercise', 'Weight']
shap_values = np.random.normal(0, 1, (100, len(shap_cols)))

cf_data = pd.DataFrame(np.random.rand(5, 5), columns=shap_cols)

radar_cats = shap_cols
before = [0.8, 0.9, 0.6, 0.5, 1.0]
after  = [0.6, 0.4, 0.3, 0.5, 0.8]


phq9_score = 17
max_phq9    = 27
fill_deg    = (phq9_score / max_phq9) * 360

# choose pale gauge color
if phq9_score < 5:
    gauge_color = 'palegreen'
elif phq9_score < 10:
    gauge_color = 'lightyellow'
elif phq9_score < 15:
    gauge_color = 'peachpuff'
else:
    gauge_color = 'lightcoral'


plt.figure(figsize=(16, 9), facecolor='white')

# 1) Top-left
ax1 = plt.subplot2grid((3, 4), (0, 0))
ax1.scatter(x, y, color='lightsteelblue')
m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x + b, color='gray', linewidth=2)
ax1.set_title('PHQ-9 Severity vs Time')

# 2) Top-middle 
ax2 = plt.subplot2grid((3, 4), (0, 1), colspan=2)
ax2.axis('off')
ax2.set_title('Top 10 Important Features', loc='left', fontsize=16, fontweight='bold', color='#333')
for i, (feat, (val, chg)) in enumerate(important_features.items()):
    col = 'palegreen' if chg > 0 else 'lightcoral'
    arr = '▲' if chg > 0 else '▼'
    ax2.text(0.05, 0.9 - i*0.07, f"{feat}: {val} {arr}", color=col, fontsize=12)

# 3) Top-right 
ax3 = plt.subplot2grid((3, 4), (0, 3))
ax3.set_aspect('equal')
ax3.axis('off')

# light-grey background circle
bg_circle = Circle((0,0), 1.0, color='whitesmoke', zorder=1)
ax3.add_patch(bg_circle)

# colored wedge
wg = Wedge((0,0), 1.0,
           theta1=90, theta2=90 + fill_deg,
           width=0.2, color=gauge_color, zorder=2)
ax3.add_patch(wg)

# center text
ax3.text(0, 0, str(phq9_score),
         ha='center', va='center',
         fontsize=24, fontweight='bold',
         color=gauge_color, zorder=3)

ax3.set_xlim(-1.2, 1.2)
ax3.set_ylim(-1.2, 1.2)
ax3.set_title('Current PHQ-9', fontsize=14, color='#333')

# Bottom-left
ax4 = plt.subplot2grid((3, 4), (1, 0))
sns.violinplot(data=pd.DataFrame(shap_values, columns=shap_cols),
               orient='h', ax=ax4, palette='pastel', inner='quartile')
ax4.set_title('SHAP Summary Plot', color='#333')

# Middle-right 
ax5 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
for idx in range(cf_data.shape[0]):
    ax5.plot(cf_data.columns, cf_data.iloc[idx],
             alpha=0.7, linewidth=2)
ax5.set_title('Counterfactuals', color='#333')

# Bottom-right
ax6 = plt.subplot2grid((3, 4), (2, 2), polar=True)
angles = np.linspace(0, 2*np.pi, len(radar_cats), endpoint=False).tolist()
angles += angles[:1]
vals_b = before + before[:1]
vals_a = after  + after[:1]

ax6.plot(angles, vals_b, color='lightcoral', marker='o', label='Before')
ax6.fill(angles, vals_b, color='lightcoral', alpha=0.25)
ax6.plot(angles, vals_a, color='lightskyblue', marker='o', label='After')
ax6.fill(angles, vals_a, color='lightskyblue', alpha=0.25)

ax6.set_thetagrids(np.degrees(angles[:-1]), radar_cats, color='#333')
ax6.set_title('Before / After', color='#333')
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
