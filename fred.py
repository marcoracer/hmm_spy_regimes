import numpy as np
import pandas as pd
import sklearn.mixture as mix
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator
import seaborn as sns
import missingno as msno
import quandl as qd

# reference:
# http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

# get fed data
f1 = 'TEDRATE' # ted spread
f2 = 'T10Y2Y' # constant maturity ten yer - 2 year
f3 = 'T10Y3M' # constant maturity 10yr - 3m

start = pd.to_datetime('2002-01-01')
end = pd.datetime.today()

data_SPY = qd.get('LSE/SPY5')
data_f1 = qd.get('FRED/TEDRATE')
data_f2 = qd.get('FRED/T10Y2Y')
data_f3 = qd.get('FRED/T10Y3M')

data = pd.concat([data_SPY['Price'], data_f1, data_f2, data_f3], axis=1, join='inner')
data.columns = ['SPY', f1, f2, f3]
data['sret'] = np.log( data['SPY']/ data['SPY'].shift(1))

print(' --- Data ---')
print(data.tail())

# quick visual inspection of the data
msno.matrix(data)

col = 'sret'
select = data.ix[:].dropna()

ft_cols = [f1, f2, f3, col]
X = select[ft_cols].values

print('\nFitting to HMM and decoding ...', end='')

model = mix.GaussianMixture(n_components=4,
	covariance_type='full',
    n_init=100,
    random_state=7).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print('done!\n')

print('Score: %.2f;\tBIC: %.2f;\tAIC:%.2f;\n' % (model.score(X), model.bic(X), model.aic(X)))

print('Means and vars of each hidden state')
for i in range(model.n_components):
	print('%d th hidden state' % i)
	print('mean = ', model.means_[i])
	print('var = ', np.diag(model.covariances_[i]))
	print()

sns.set(font_scale=1.25)
style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
	'font.family':u'courier prime code', 'legend.frameon': True}
sns.set_style('white', style_kwds)

fig, axs = plt.subplots(model.n_components, sharex=True, figsize=(12,9))
colors = cm.rainbow(np.linspace(0, 1, model.n_components))

for i, (ax, color) in enumerate(zip(axs, colors)):
	# Use fancy indexing to plot data in each state.
	mask = hidden_states == i
	ax.plot_date(select.index.values[mask],
	             select[col].values[mask], '.-', c=color)
	ax.set_title('%d th hidden state' % i, fontsize=16, fontweight='demi')

	# Format the ticks.
	ax.xaxis.set_major_locator(YearLocator())
	ax.xaxis.set_minor_locator(MonthLocator())
	sns.despine(offset=10)

plt.tight_layout()
plt.show()
# fig.savefig('Hidden Markov (Mixture) Model_Regime Subplots.png')


sns.set(font_scale=1.5)
states = (pd.DataFrame(hidden_states, columns=['states'], index=select.index)
	.join(select, how='inner')
	.assign(mkt_cret=select.sret.cumsum())
	.reset_index(drop=False)
	.rename(columns={'index':'Date'}))

print(' --- States ---')
print(states.tail())

sns.set_style('white', style_kwds)
order = [0, 1, 2]
fg = sns.FacetGrid(data=states, hue='states', hue_order=order,
	palette=colors, aspect=1.31, size=12)
fg.map(plt.scatter, 'Date', 'SPY', alpha=0.8).add_legend()
sns.despine(offset=10)
fg.fig.suptitle('Historical SPY Regimes', fontsize=24, fontweight='demi')
plt.tight_layout()
plt.show()
# fg.savefig('Hidden Markov (Mixture) Model_SPY Regimes.png')