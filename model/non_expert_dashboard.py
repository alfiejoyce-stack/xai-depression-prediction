import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.patches import Wedge, Circle


sns.set_palette('pastel')
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['text.color']     = '#333'


np.random.seed(0)

# PHQ-9 vs Time
times     = np.linspace(2, 12, 15)
phq9_vals = 2.2 * times + np.random.normal(0, 3, size=times.shape)


important = {
    'Sleep (hrs/night)':  (6.0, -1),
    'Exercise (hrs/wk)':  (1.5, -1),
    'Social (hrs/day)':   (0.5, -1),
    'Nutrition (score)':  (4.0,  1),
    'Age (years)':        (30,   1)
}


features = list(important.keys())
shap_vals = np.random.normal(0, 1, (100, len(features)))
shap_imp = pd.DataFrame({
    'Feature':    features,
    'Importance': np.mean(np.abs(shap_vals), axis=0)
})
shap_imp['Color'] = shap_imp['Importance'].apply(lambda x: 'lightgreen' if x>0 else 'lightcoral')


X_dt = np.random.rand(200, 3)

y_dt = ((X_dt[:,0] + X_dt[:,1] + X_dt[:,2]) > 1.5).astype(int)

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_dt, y_dt)


counterfactuals_list = [
    np.array([0.7, 0.2, 0.1]),
    np.array([0.3, 0.8, 0.5])
]

phq9_score = 11
max_phq9    = 27
fill_deg    = (phq9_score / max_phq9) * 360
gauge_color = 'lightcoral' if phq9_score > 10 else 'palegreen'

fig = plt.figure(figsize=(16,10), facecolor='white')
gs  = fig.add_gridspec(3,4, wspace=0.6, hspace=0.6)

ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(times, phq9_vals, color='lightsteelblue')
m, b = np.polyfit(times, phq9_vals, 1)
ax1.plot(times, m*times + b, color='gray', linewidth=2)
ax1.set_title('PHQ-9 Severity vs Time')

ax2 = fig.add_subplot(gs[0,1:3])
ax2.axis('off')
ax2.set_title('Key Factors Affecting Mood', fontsize=16, fontweight='bold')
for i,(feat,(val,chg)) in enumerate(important.items()):
    col   = 'palegreen' if chg>0 else 'lightcoral'
    arrow = '▲' if chg>0 else '▼'
    ax2.text(0.02, 0.9 - i*0.12, f"{feat}: {val} {arrow}", color=col, fontsize=13)


ax3 = fig.add_subplot(gs[0,3])
ax3.set_aspect('equal'); ax3.axis('off')
ax3.add_patch(Circle((0,0),1.0, color='whitesmoke', zorder=1))
ax3.add_patch(Wedge((0,0),1.0, theta1=90, theta2=90+fill_deg,
                    width=0.2, color=gauge_color, zorder=2))
ax3.text(0,0,str(phq9_score), ha='center', va='center',
         fontsize=24, fontweight='bold', color=gauge_color, zorder=3)
ax3.set_xlim(-1.2,1.2); ax3.set_ylim(-1.2,1.2)
ax3.set_title('Current PHQ-9', fontsize=14)


ax4 = fig.add_subplot(gs[1:,0:2])
bars = ax4.barh(shap_imp['Feature'], shap_imp['Importance'],
                color=shap_imp['Color'], edgecolor='white')
ax4.invert_yaxis()
ax4.set_title('Feature Importance (SHAP)', fontsize=14)

ax5 = fig.add_subplot(gs[1,2:])
plot_tree(tree,
          feature_names=['Sleep (norm)', 'Exercise (norm)', 'Social (norm)'],
          filled=False,
          impurity=False,    
          node_ids=False,    
          proportion=False,  
          class_names=None,  
          label='none',      
          ax=ax5)

for txt in ax5.texts:
    text = txt.get_text().split('\n')[0]  
    if 'class' in text or 'samples' in text:
        text = text.split('class')[0].strip()
    txt.set_text(text)

ax5.set_title('Decision Tree (Feature Changes for Target PHQ-9)', fontsize=14)


ax6 = fig.add_subplot(gs[2,2:])
ax6.axis('off')
ax6.text(0,0.5,
    "To improve your PHQ-9 score:\n\n"
    "• Prioritize getting 7–8 hours of sleep each night.\n"
    "• Incorporate at least 150 minutes of exercise per week.\n"
    "• Engage in social activities for at least 30 min/day.\n"
    "• Eat a balanced diet with plenty of fruits and vegetables.",
    ha='left', va='center', fontsize=13)

plt.tight_layout()
plt.show()
