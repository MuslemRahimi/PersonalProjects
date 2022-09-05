import pandas as pd 
import time
import matplotlib.pyplot as plt
import numpy as np


def monthly_spending(month):

	df = pd.read_csv('data/'+month+'_2022.csv',sep=',')


	budget = []
	name = []
	spending = pd.DataFrame()
	gehalt = 0

	name_list = ['Gehalt','Miete','Amazon','Lidl','Aldi','Kaufland','Paypal','dm','Rossmann','TEDi',\
				'TradeRepublic','Action','Waschsalon','HandyVertrag','SDK','Gehalt']
	for i in range(len(df['Name Zahlungsbeteiligter'])):
		if 'LBV' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Gehalt','Euro': df['Betrag'][i]},ignore_index=True)
			gehalt = df['Betrag'][i]

		elif 'AXA' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Miete','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'AMAZON' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Amazon','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'LIDL' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Lidl','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'ALDI' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Aldi','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'KAUFLAND' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Kaufland','Euro': df['Betrag'][i]},ignore_index=True)
		
		elif 'Lastschrift' in df['Name Zahlungsbeteiligter'][i]:
			if 'WASCHSALON SIEGEN' in str(df['Verwendungszweck'][i]):
				spending=spending.append({'Name': 'Waschsalon','Euro': df['Betrag'][i]},ignore_index=True)
			else:
				spending=spending.append({'Name': 'Sonstige','Euro': df['Betrag'][i]},ignore_index=True)
		
		elif 'PAYPAL' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Paypal','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'DM' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'dm','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'ROSSMANN' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'Rossmann','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'TEDi' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'TEDi','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'Muslem' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'TradeRepublic','Euro': df['Betrag'][i]},ignore_index=True)
		
		elif 'Action' in str(df['Verwendungszweck'][i]):
			spending=spending.append({'Name': 'Action','Euro': df['Betrag'][i]},ignore_index=True)
	
		elif 'Drillisch' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'HandyVertrag','Euro': df['Betrag'][i]},ignore_index=True)
		elif 'Sueddeutsche' in df['Name Zahlungsbeteiligter'][i]:
			spending=spending.append({'Name': 'SDK','Euro': df['Betrag'][i]},ignore_index=True)
		else:
			spending=spending.append({'Name': 'Sonstige','Euro': df['Betrag'][i]},ignore_index=True)
			
	spending = spending.groupby(['Name']).sum()
	spending = spending.reset_index()

	#Add names that are not included because 'Euro' is zero
	for i in name_list:
		if i in [x for x in spending['Name']]:
			pass
		else:
			spending=spending.append({'Name': i,'Euro': 0},ignore_index=True)

	print(spending)
	total_spending = spending['Euro'].sum()

	#Combine into categories such as Lebensmittel, Fixkosten, OnlineBestellung, Haushaltsmittel, Investment
	def categories(data):
		cat = pd.DataFrame()
		for i in range(len(spending['Name'])):
			if spending['Name'][i] == 'Amazon' or spending['Name'][i] == 'Paypal':
				cat=cat.append({'cat': 'OnlineBestellung','Euro': spending['Euro'][i]},ignore_index=True)
			elif spending['Name'][i] == 'Miete' or spending['Name'][i] == 'HandyVertrag' \
				or spending['Name'][i] == 'SDK':
				cat=cat.append({'cat': 'Fixkosten','Euro': spending['Euro'][i]},ignore_index=True)
			elif spending['Name'][i] == 'TradeRepublic':
				cat=cat.append({'cat': 'Investment','Euro': spending['Euro'][i]},ignore_index=True)
			elif spending['Name'][i] == 'dm' or spending['Name'][i] == 'Rossmann' \
				or spending['Name'][i] == 'TEDi' or spending['Name'][i] == 'Waschsalon' \
				or spending['Name'][i] == 'Action':
				cat=cat.append({'cat': 'Haushaltskosten','Euro': spending['Euro'][i]},ignore_index=True)

			elif spending['Name'][i] == 'Lidl' or spending['Name'][i] == 'Kaufland' \
				or spending['Name'][i] == 'Aldi':
				cat=cat.append({'cat': 'Lebensmittel','Euro': spending['Euro'][i]},ignore_index=True)
			elif spending['Name'][i] == 'Sonstige':
				cat=cat.append({'cat': 'Sonstige','Euro': spending['Euro'][i]},ignore_index=True)
		cat = cat.groupby(['cat']).sum()
		cat = cat.reset_index()
		return cat

	cat= categories(spending)
	print(cat)
	print("Total saving of %s: %s EUR"% (month, np.around(total_spending,3)))
	print("#========================#")


	return cat, gehalt

def plot_compare(data):
	pass
	

month = ['June','July','August']
june, income_june = monthly_spending('June')
july,income_july = monthly_spending('July')
august,income_august = monthly_spending('August')

#======================================#

x = np.arange(len(june['cat']))
x_income = np.arange(1)

width = 0.25
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

fig, (ax1, ax2) = plt.subplots(2)
ax1.bar(x +0, round(june['Euro'],3), width,  label='June')
ax1.bar(x +0.25, round(july['Euro'],3), width, label='July')
ax1.bar(x +0.5, round(august['Euro'],3), width, label='August')

ax2.bar(x_income +0, round(income_june,3), 0.1,  label='June')
ax2.bar(x_income +0.25, round(income_july,3), 0.1, label='July')
ax2.bar(x_income +0.5, round(income_august,3), 0.1, label='August')

ax1.set_ylabel('Euro')
ax1.set_title('Spending')
ax1.set_xticks(x)
ax1.set_xticklabels(june['cat'])
ax1.legend()

fig.tight_layout()
plt.show()
