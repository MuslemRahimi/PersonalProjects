import numpy as np 
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


haus_preis = 500E3
#Maklerprovision, Grunderwerbsteuer, Notarkosten, Grundbucheintrag
nebenkosten = haus_preis * 0.10
eigenkaptial = haus_preis*0.20

nettodarlehen = haus_preis+nebenkosten - eigenkaptial

print('Nettodarlehen: %s EUR' % nettodarlehen)
eff_Jahreszins = 2.99*10**(-2)
monatsrate = 2044 #Euro
jahre = 20
restschuld = nettodarlehen

liste_restschuld = []
liste_jahre = []
j= 1
while restschuld > 0:
	Jahres_zins = 0
	Jahres_tilgung = 0
	Jahres_rate = 0
	for m in range(1,13):
		zins = restschuld*eff_Jahreszins/12.0
		tilgung = monatsrate - eff_Jahreszins*restschuld/12.0
		raten = zins+tilgung
		restschuld = restschuld - raten

		Jahres_tilgung += tilgung
		Jahres_zins += zins
		Jahres_rate += raten

		print("Monat %s: Monatsrate: %s: Zinsen: %s: Tilgung: %s:" %(m, round(raten,2), round(zins,2), round(tilgung,2)) )

	liste_restschuld.append(restschuld)
	liste_jahre.append(j)
	print("Jahr %s: Restschuld: %s: Raten %s: Zinsen: %s: Tilgung: %s" \
		  %(j, round(restschuld,2), round(Jahres_rate,2), round(Jahres_zins,2), round(Jahres_tilgung,2) ) )
	print("#========================================#")
	j +=1

fig, ax = plt.subplots()


plt.xlim(1,jahre)
plt.ylim(0,nettodarlehen)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


plt.tick_params(axis='both', which='major', labelsize=15,length=9,direction='in')
plt.tick_params(axis='both', which='minor', labelsize=9,length=5,direction='in')


ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax.yaxis.set_major_locator(ticker.MultipleLocator(100000))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50000))


ax.plot(liste_jahre, liste_restschuld)
plt.xlabel('Jahre')
plt.ylabel('Euro')
plt.grid(True)
plt.show()