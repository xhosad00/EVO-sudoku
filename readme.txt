Návod na spuštění programu

vytvoření prostředí:
    Windows:
        python -m venv venv
        venv\Scripts\activate
        pip install -r requirements.txt

    Linux: (netestováno)
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt


Konfigurace testovacího běhu je převážně v globálních konstantách. Ve funkci main() se nastaví hodnoty dle zkoumaného atributu.
Jednotlivé měření byly provedeny ručně. Zde je příkladné nastavení na zkoumání obtížnosti:
configName = 'DIFFICULTY'
configValues = list([0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 0.95])