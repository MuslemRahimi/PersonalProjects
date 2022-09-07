import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as ticker

 
def berechne_jaehrliche_Annuitaet(kreditsumme, nominalzins_prozent, tilgungssatz_prozent):
    """ Berechnet die _jährliche_ Annuität.
        Jährliche_Rate = (nominalzins + tilgungssatz) * Kreditsumme
        Quelle: https://de.wikipedia.org/wiki/Annuit%C3%A4tendarlehen
    """ 
 
    zinssatz = nominalzins_prozent / 100
    tilgung = tilgungssatz_prozent / 100
    return round(kreditsumme * (zinssatz + tilgung), 2)
 
 
def berechne_monatliche_Annuitaet(kreditsumme, nominalzins_prozent, tilgungssatz_prozent):
    """ Berechnet die _monatliche_ Annuität.
        Jährliche_Rate = (nominalzins + tilgungssatz) * Kreditsumme
        Monatliche_Rate = Jährliche_Rate / 12
    """ 
 
    zinssatz = nominalzins_prozent / 100
    tilgung = tilgungssatz_prozent / 100
    return round(kreditsumme * (zinssatz + tilgung) / 12, 2)
 
def tilgungsplan_df(kreditsumme, nominalzins_prozent, tilgungssatz_prozent, sondert, wartezeit):
    """ 
        Gibt DataFrame der monatlichen Tilgungen zurück
 
        "monate" für wieviele Monate wird der Tilgungsplan erstellt
        "sondert" Betrag der jährlichen Sondertilgung
        "wartezeit" Anzahl der Jahre ohne Sondertilgung
    """
 
    df = pd.DataFrame()
    df_jahr = pd.DataFrame()
    restschuld = kreditsumme # Am Anfang entspricht die Restschuld der Kreditsumme
    zinssatz = nominalzins_prozent / 100
    tilgung = tilgungssatz_prozent / 100
 
    annuitaet = berechne_monatliche_Annuitaet(kreditsumme, nominalzins_prozent, tilgungssatz_prozent)
    zinsen = 0
    jahr = 0
    monat = 0
    while restschuld > 0:
        jahr +=1
        for m in range(1,13):
            if restschuld == 0:
                break
            monat = monat+ 1
            # Split der Annuität in ihre Komponenten Zinslast und Tilgung
            zinsen = restschuld * zinssatz / 12 
            # Wenn Restschuld kleiner Annuität, dann wird die komplette 
            # Restschuld getilgt
            tilgung = restschuld if restschuld < annuitaet else annuitaet - zinsen    
     
            anfangsschuld = restschuld
            #jahr = ((j-1) // 12) + 1 # in welchem Monat befinden wir uns
            
            # Sondertilgungen im Dezember eines Jahres, wenn wir 
            # nicht in der Wartezeit sind

            if monat % 12 == 0 and anfangsschuld > 0 and jahr > wartezeit:
                sondertilgung = sondert
            else:
                sondertilgung = 0
     
            # Restschuld_neu = Restschuld_alt minus Tilgung minus Sondertilgung
            restschuld = restschuld - tilgung - sondertilgung
     
            # Dataframe befüllen
            '''
            df = df.append({'Monat': m, 'Jahr': jahr,'Anfangsschuld': anfangsschuld, 
            'Zinsen':zinsen, 'Tilgung': tilgung, 'Sondertilgung': sondertilgung,
            'Restschuld': restschuld}, ignore_index=True)    
            '''
            df = df.append({'Monat': monat, 'Jahr': jahr,'Anfangsschuld': anfangsschuld, 
            'Zinsen':zinsen, 'Tilgung': tilgung,'Restschuld': restschuld}, ignore_index=True) 

          
    # Indikatorspalte, "1" wenn der Kredit noch nicht abbezahlt ist, sonst "0"
    df['Indikator'] = np.where(df['Anfangsschuld']>0, 1, 0)
    # Umsortieren der Spalten
    df = df[['Monat', 'Jahr', 'Anfangsschuld', 'Zinsen', 'Tilgung', 'Restschuld', 'Indikator']]
 
    # Runden auf 2 Nachkommastellen
    for i in ['Anfangsschuld', 'Zinsen', 'Tilgung', 'Restschuld']:
        df[i] = df[i].apply(lambda x: round(x, 2))    
 
    # Monat als Index nutzen
    #df.set_index('Monat', inplace=True)
    return df


if __name__ == "__main__":
    # Setze die Parameter fest für die Tilgungsplan
    Kaufpreis = 500E3
    Nebenkosten = Kaufpreis*0.10
    Eigenkapital = Kaufpreis*0.20
    kreditsumme = Kaufpreis+Nebenkosten-Eigenkapital
    nominalzins_prozent= 3.36
    tilgungssatz_prozent = 2.0
    #Sollzinsbindung in Monaten
    sollzinsbindung = 15*12

    tilgungsplan = tilgungsplan_df(kreditsumme, nominalzins_prozent, tilgungssatz_prozent, 0, 0)
    print(tilgungsplan)
    print('#=========================================#')
    print('Kaufpreis: %s EUR: Nebenkosten: %s EUR: Eigenkaptial: %s: EUR' % (Kaufpreis,Nebenkosten,Eigenkapital))
    print('Kreditsumme: %s EUR: Sollzins: %s : Tilgungssatz: %s:'%(kreditsumme,nominalzins_prozent,tilgungssatz_prozent))
    print(berechne_jaehrliche_Annuitaet(kreditsumme, nominalzins_prozent, tilgungssatz_prozent), 'jährliche Annuität')
    print(berechne_monatliche_Annuitaet(kreditsumme, nominalzins_prozent, tilgungssatz_prozent), 'Monatsrate')
    #Wie lange läuft der Kredit
    Monats_laufzeit = round(tilgungsplan['Indikator'].sum(),1)
    print('Gesamtlaufzeit:', Monats_laufzeit, 'Monate')
    print('Gesamtlaufzeit:', round(Monats_laufzeit/12,1), 'Jahre')
    print('#=========================================#')

    
    fig, ax = plt.subplots()

    liste_monate = [int(m) for m in range(1,Monats_laufzeit+1)]
    liste_monate_sollzinsbindung = [int(m) for m in range(1,sollzinsbindung)]
    liste_monate_restlaufzeit = [int(m) for m in range(sollzinsbindung, Monats_laufzeit)]
    liste_restschuld = [r for r in tilgungsplan['Restschuld']]

    plt.xlim(1,liste_monate[-1])
    plt.ylim(0,kreditsumme)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


    plt.tick_params(axis='both', which='major', labelsize=15,length=9,direction='in')
    plt.tick_params(axis='both', which='minor', labelsize=9,length=5,direction='in')


    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(12))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(150000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(75000))



    ax.fill_between(liste_monate_sollzinsbindung, liste_restschuld[0:liste_monate_sollzinsbindung[-1]],\
                    0, alpha=0.3,color='green',label='Sollzinsbindung von 15 Jahren')
    ax.fill_between(liste_monate_restlaufzeit, liste_restschuld[liste_monate_sollzinsbindung[-1]:-1],\
                    0, alpha=0.3,color='orange',label='Gleichbleibende Sollzinsbindung')


    plt.axvline(x=15*12,color='blue', lw=6, alpha=0.5, label='Ende der Sollzinsbindung')
    plt.xlabel('Monate',fontsize=18)
    plt.ylabel('Euro',fontsize=18)

    ax.legend(shadow=True,fontsize=15)
    plt.show()
