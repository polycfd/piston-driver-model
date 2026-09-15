import matplotlib.pyplot as plt

# ============================================================
# PLOTTING STYLE
# ============================================================

def new_figure(figsize=(18, 9)):
    return plt.figure(figsize=figsize)

def style_axes(ax, spine_width=2, tick_labelsize=35, tick_bold=True):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    for label in ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'):
        if tick_bold:
            label.set_fontweight('bold')
        label.set_fontsize(tick_labelsize)

def style_legend(legend, bold=True, fontsize=35):
    for text in legend.get_texts():
        if bold:
            text.set_fontweight('bold')
        text.set_fontsize(fontsize)

def style_grid(ax=None, linestyle='--', alpha=0.6):
    if ax is None:
        ax = plt.gca()
    ax.grid(True, linestyle=linestyle, alpha=alpha)

def savefig(fig, filename, folder='figures'):
    fig.savefig(f'{folder}/{filename}.pdf', bbox_inches='tight', pad_inches=0.05, dpi=300)