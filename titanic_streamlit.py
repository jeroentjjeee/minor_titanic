import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import uuid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Modeling & evaluatie
# import sklearn

# Hiermee laat je alle kolommen standaard zien
pd.set_option('display.max_columns', None)
if 'df' not in st.session_state:

    train = pd.read_csv("train.csv")

    train = train.drop(['Cabin', 'Ticket'], axis = 1)
    train['Nlength'] = train['Name'].str.len()
    train["status"] = np.where(train["Survived"] == 1, "Overleeft", "Overleden")
    train['Famsize'] = train['SibSp'] + train['Parch']
    dfna = train.isna().sum().reset_index()
    dfna.columns = ['Kolom', 'Aantal missende waarden']
    dfna = dfna.set_index('Kolom')
    dfna = dfna.sort_values('Aantal missende waarden', ascending = False)
    st.session_state.dfna = dfna
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
    st.session_state.data = train
df_zonder = st.session_state.data.drop(columns = ['Famsize', 'Nlength'])
df = st.session_state.data
dfna = st.session_state.dfna


def plot_histogram(df, kolommen):
    kolom = st.selectbox("Kies een variabele om te visualiseren:", kolommen)

    
    titel_mapping = {
        'status': 'Verdeling overlevingsstatus',
        'Pclass': 'Verdeling klasses',
        'Sex': 'Verdeling genders',
        'SibSp': 'Verdeling aantal broers, zussen of echtgenoten aan boord',
        'Parch': 'Verdeling aantal ouders of kinderen aan boord',
        'Fare': 'Verdeling Fare',
        'Embarked': 'Verdeling opstaplocatie',
        'Age': 'Verdeling leeftijd',
        'Famsize': 'Verdeling familieleden aan boord',
        'Nlength': 'Verdeling lengte van de namen van passagiers'
    }

    xlabel_mapping = {
        'status': 'Overlevingsstatus',
        'Pclass': 'Klasse',
        'Sex': 'Gender',
        'SibSp': 'Aantal broers, zussen of echtgenoten',
        'Parch': 'Aantal ouders of kinderen',
        'Fare': 'Fare',
        'Embarked': 'Opstaplocatie',
        'Age': 'Leeftijd',
        'Famsize': 'Aantal familieleden',
        'Nlength': 'Lengte naam'
    }

    kleuren_dict = {
    'status': {
        'Overleeft': 'limegreen',
        'Overleden': 'red'
    },
    'Pclass': {
        1: '#1f77b4',   # blauw
        2: '#ff7f0e',   # oranje
        3: '#2ca02c'    # groen
    },
    'Embarked': {
        'C': '#1f77b4',         # Cherbourg
        'Q': '#9467bd',         # Queenstown
        'S': '#ff7f0e'          # Southampton
    },
    'Sex': {
        'male': '#1f77b4',      # blauw
        'female': '#e377c2'     # roze/paars
    }
}

    
    if kolom == 'Fare':
        st.write('In de slider kan de fare range gekozen worden.')
        waarde = st.slider('Filter de fare:', int(df[kolom].min()), 265, value = (int(df[kolom].min()), 265), step = 1)
        data = df[(df['Fare'] >= waarde[0]) & (df['Fare'] <= waarde[1])]
        extra_args = {'nbins': 50}
    else:
        data = df
        extra_args = {}
    
    fig = px.histogram(
        data,
        x=kolom,
        color_discrete_sequence=px.colors.qualitative.Set1,
        **extra_args
    )

    
    fig.update_layout(
        title = {'text': titel_mapping.get(kolom, kolom),
                'x':0.5,
                'xanchor': 'center' },
        
        xaxis_title=xlabel_mapping.get(kolom, kolom),
        yaxis_title='Aantal'
        )
    fig.update_traces(marker_line_color='black', marker_line_width=1)

    
    if kolom == 'Embarked':
        fig.update_xaxes(
            categoryorder='array',
            tickvals=['C', 'Q', 'S'],
            ticktext=['Cherbourg', 'Queenstown', 'Southampton']
        )

    
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{kolom}_{uuid.uuid4()}")

def bereken_percentages(df, groep_col):
    """
     Hulpfunctie om per categorie het percentage per status te berekenen.
    """
    df_counts = df.groupby([groep_col, 'status']).size().reset_index(name='Aantal')
    # Totale aantallen per categorie berekenen
    totaal = df_counts.groupby(groep_col)['Aantal'].transform('sum')
    # Percentage per categorie
    df_counts['Percentage'] = (df_counts['Aantal'] / totaal * 100).round(2)
    return df_counts



keuze = st.sidebar.radio('Kies een pagina', ['Voorpagina', 'Variabelen', 'Visualisaties', 'Voorspelling'])
if keuze == 'Voorpagina':
    st.set_page_config(layout="wide")
    tab1, tab2, tab3 = st.tabs(['Uitleg dashboard','Verschillen t.o.v. vorige keer', 'Eerste keer'])
    with tab1:
        st.title('Dashboard Titanic')
        st.write(
        "In dit dashboard bekijken en interperteren we data van de titanic om op basis van deze data te kunnen voorspellen of iemand de titanic zal hebben overleeft.")
        st.markdown("Dit dashboard kunt u als volgt navigeren:  \n" \
        "In de sidebar is de pagina te kiezen, daarnaast zijn in het tabmenu subpaginas te bekijken.")
        st.markdown(
        """
        In dit dashboard behandelen we de volgende onderwerpen:  
        
        - **De variabelen**  
        - **De visualisaties van de variabelen**  
        - **Ons voorspellingsmodel**
        """)

        st.markdown(
            "Dit is de tweede keer dat wij naar deze data kijken. Om te kijken wat wij anders hebben gedaan, kan gekeken worden naar de subpaginas verchil en eerste poging."
        )
    with tab2:
        st.title('Verschillen ten opzichte van de vorige keer')
        st.markdown("""Op deze pagina staan de verschillen tussen de eerste een tweede keer dat deze data behandeld wordt.""")
        st.header('Data')
        st.markdown("""Bij de **eerste keer** zijn er **geen** extra variabelen toegevoegd en kolommen met missende waarden zijn weggelaten.    
            Bij deze poging zijn er 2 extra variabelen toegevoegd:  
            - **Naamlengte**  
            - **Familiegrootte**  
            Missende waarden zijn opgevuld door de mediaan of de modus.  """)
        st.header('Visualisaties')
        st.markdown("""Eerste keer:  
                    - De eerste keer waren de visualisaties gemaakt met maplotlib en seaborn.  
                    - Alleen visualisaties op variabelen die in het model zijn meegenomen.
                    - Er was feedback dat er meer geen log schaal gebruikt werd waar het nodig was bij visualisaties.  
                    - Er was feedback dat percentages beter gebruikt konden worden dan absolute aantallen bij visualisaties.  
                    Tweede keer:  
                    - De visualisaties zijn voornaamlijk met plotly.express gemaakt.    
                    - Er zijn visualisaties van elke variabele gemaakt, zelfs als deze niet in het model zitten.  
                    - Nu wordt er rekening gehouden met log schalen en percentages in plaats van absolute aantallen
                    """)

        st.header('Modellen')
        st.markdown("""Eerste keer:  
                    - Het model is gemaakt door keuzes op basis van visualisaties.  
                    Tweede keer:  
                    - Er wordt gebruik gemaakt van een complexer model.""")
    with tab3:
        st.title('Eerste keer')
        st.header('1. Dataverkenning')
        st.dataframe(df_zonder.head())
        st.markdown("""
        Deze variabelen hebben de volgende betekenis:

        - **Survived**: 1 als de persoon overleeft en anders 0.  
        - **Pclass**: Klasse waarin de passagier heeft gereisd.  
        - **Name**: Naam van de passagier.  
        - **Sex**: Gender van de passagier.  
        - **Age**: Leeftijd van de passagier.  
        - **SibSp**: Aantal broers, zussen of echtgenoten aan boord.  
        - **Parch**: Aantal ouders of kinderen aan boord.  
        - **Embarked**: Opstaplocatie van de passagier — S = Southampton, Q = Queenstown, C = Cherbourg.  
        - **status**: 'Overleefd' als de passagier overleeft en anders 'overleden'.  
        """)

        st.header('1.1 Missende waarden')
        st.dataframe(dfna[dfna['Aantal missende waarden'] > 0])
        st.write('Om deze missende waarden op te lossen hebben wij de desbetreffende waarnemingen uit de dataset verwijderd.')

        st.header('2. Visualisaties')
        st.write('Hieronder de visualisaties')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_palette(["r", "black"])

        
        sns.histplot(
            data=train,
            x="Age",            
            hue="Survived",     
            alpha=0.6,
            multiple="dodge",
            ax=ax
        )

        
        ax.set_title("Verdeling van de leeftijden", fontsize=14)
        ax.set_xlabel("Leeftijd")
        ax.set_ylabel("Aantal")

        st.pyplot(fig)

        sns.set_style("whitegrid")
        sns.set_palette(["b","y"])

        p = sns.relplot(data=train, x="Age", y="Fare", kind='scatter', hue="Survived", alpha=0.5)
        p.fig.suptitle("Leeftijd vs Kosten", fontsize=14)
        p.set(xlabel="Leeftijd", ylabel="Kosten")

        st.pyplot(p.fig)   
        plt.close()


        
        sns.set_palette(["r","purple"])

        g = sns.catplot(data=train, x="Sex", y="Survived", kind="bar", hue="Sex")
        g.fig.suptitle("Overlevingskans per geslacht", fontsize=14)
        g.set(xlabel="Gender", ylabel="Overlevingskans")

        st.pyplot(g.fig)
        plt.close()
        sns.set_style("whitegrid")
        sns.set_palette(["r", "purple"])

        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=train, x="Sex", y="Age", hue="Sex", ax=ax)

        
        ax.set_title("Gender vs Leeftijd", fontsize=14)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Leeftijd")

        
        st.pyplot(fig)
        plt.close(fig)


        
        palette = ["lime", "r"]
        sns.set_palette(palette)

        f = sns.catplot(kind="count", x="Pclass", hue="Survived", data=train, palette=palette)
        f.fig.suptitle("Hoeveelheid passagiers per klasse met overlevingsstatus", fontsize=14)
        f.set(xlabel="Klasse", ylabel="Aantal")

        st.pyplot(f.fig)
        plt.close()

        st.header('3. Model')
        st.markdown("""Het model is gemaakt door te kijken naar welke variabelen en aspecten daarvan correleren met de overlevingskans.  
                    Dit zijn de klasse, gender en leeftijd variabelen.  
                    Bij deze variabelen kijken we naar de volgende aspecten:  
                    - **Leeftijd**: Als de passagier Jonger dan 10 dan overleven ze.  
                    - **Klasse**: Als passagiers met klasse 3 reizen overleven ze het niet.    
                    - **Gender**: Als de passagier een vrouw is overleven ze.  
                    In deze volgorde werkt het model en dit geeft een score op kaggle van 77.03%""")




elif keuze == 'Variabelen':
    st.set_page_config(layout="wide")
    st.title('Uitleg variabelen')
    st.header('1. Data verkenning')
    st.dataframe(df_zonder.head(), use_container_width=True)
    st.markdown("""
    Deze variabelen hebben de volgende betekenis:

    - **Survived**: 1 als de persoon overleeft en anders 0.  
    - **Pclass**: Klasse waarin de passagier heeft gereisd.  
    - **Name**: Naam van de passagier.  
    - **Sex**: Gender van de passagier.  
    - **Age**: Leeftijd van de passagier.  
    - **SibSp**: Aantal broers, zussen of echtgenoten aan boord.  
    - **Parch**: Aantal ouders of kinderen aan boord.  
    - **Embarked**: Opstaplocatie van de passagier — S = Southampton, Q = Queenstown, C = Cherbourg.  
    - **status**: 'Overleefd' als de passagier overleeft en anders 'overleden'.  
    """)

    st.header('1.1 Data opvulling')
    st.write('Uiteraard heeft de data ook een aantal missende waarden:')
    st.dataframe(dfna[dfna['Aantal missende waarden'] > 0])
    st.markdown("""
    Deze missende waarden zijn we als volgt opgelost:
        
    - **Age**: Deze missende waarden zijn vervangen met de mediaan.
    - **Embarked**: Deze missende waarden zijn vervangen met de modus.""")

    st.header('1.2 Nieuwe variabelen toevoegen')
    st.markdown("""Naast te bestaande kolommen zijn er ook nieuwe variabelen toegevoegd.    
    - **Nlength**: De lengte van de naam (Aantal letters/karakters).  
    - **Famsize**: Aantal familieleden aan boord""")
    st.dataframe(df[['Famsize', 'Nlength']].head())

    st.header('1.3 Verdeling van de variabelen')
    st.markdown("""
    Hieronder kun je de verdeling van de variabelen zien.
    In het dropdown menu kan de variabele die je wil zien gekozen worden.""")
    plot_histogram(df, ['status', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age', 'Famsize'])
    


        
elif keuze == 'Visualisaties':
    st.set_page_config(layout="wide")
    tab1, tab2, tab3 = st.tabs(['Variabele met overlevingskans','Andere viasualisaties','Kaart Titanic'])
    with tab1:
        st.title('2. Visualisaties')
        st.header('2.1 Variabelen gecombineerd met de overlevingskans')
        palette = {
    'Overleden': "red",      # Niet overleefd
    'Overleeft': "lime"      # Overleefd
    }

    
        palette = {
            'Overleden': "red",      
            'Overleeft': "lime"      
        }
        kleuren = ["lime", "red"]

        # 1️⃣ Passagiers per klasse
        df_pclass = bereken_percentages(df, 'Pclass')
        fig1 = px.bar(
            df_pclass,
            x="Pclass",
            y="Percentage",
            color="status",
            color_discrete_sequence=kleuren,
            category_orders={"status": ["Overleeft", "Overleden"]},
            barmode="group",
            opacity=1.0
        )
        fig1.update_layout(title = {'text':'Overlevings en overlijdingskans passagiers per klasse ','x': 0.5, 'xanchor': 'center' },
                           xaxis_title="Klasse", yaxis_title="Percentage (%)")
        st.plotly_chart(fig1)

        # 2️⃣ Passagiers per geslacht
        df_sex = bereken_percentages(df, 'Sex')
        fig2 = px.bar(
            df_sex,
            x="Sex",
            y="Percentage",
            color="status",
            color_discrete_sequence=kleuren,
            category_orders={"status": ["Overleeft", "Overleden"]},
            barmode="group",
            opacity=1.0
        )
        fig2.update_layout(title = {'text':'Overlevings en overlijdingskans passagiers per geslacht','x': 0.5, 'xanchor': 'center' },
                           xaxis_title="Geslacht", yaxis_title="Percentage (%)")
        st.plotly_chart(fig2)

        # 3️⃣ Passagiers per opstaplocatie
        df_embarked = bereken_percentages(df, 'Embarked')
        fig3 = px.bar(
            df_embarked,
            x="Embarked",
            y="Percentage",
            color="status",
            color_discrete_sequence=kleuren,
            category_orders={"status": ["Overleeft", "Overleden"]},
            barmode="group",
            opacity=1.0
        )
        fig3.update_xaxes(
        tickvals=["C", "Q", "S"],
        ticktext=["Cherbourg", "Queenstown", "Southampton"]
        )

        fig3.update_layout(title = {'text':'Overlevings en overlijdingskans passagiers per opstaplocatie','x': 0.5, 'xanchor': 'center' },
                           xaxis_title="Opstap locatie", yaxis_title="Percentage (%)")
        st.plotly_chart(fig3)

        # 4️⃣ Passagiers per aantal broers/zussen
        df_sibsp = bereken_percentages(df, 'SibSp')
        fig4 = px.bar(
            df_sibsp,
            x="SibSp",
            y="Percentage",
            color="status",
            color_discrete_sequence=kleuren,
            category_orders={"status": ["Overleeft", "Overleden"]},
            barmode="group",
            opacity=1.0
        )
        fig4.update_layout(title = {'text':'Overlevings en overlijdingskans passagiers per aantal broers en zussen aan boord','x': 0.5, 'xanchor': 'center' },
                           xaxis_title="Aantal broers en zussen", yaxis_title="Percentage (%)")
        st.plotly_chart(fig4)

        # 5️⃣ Passagiers per aantal ouders/kinderen
        df_parch = bereken_percentages(df, 'Parch')
        fig5 = px.bar(
            df_parch,
            x="Parch",
            y="Percentage",
            color="status",
            color_discrete_sequence=kleuren,
            category_orders={"status": ["Overleeft", "Overleden"]},
            title="Overlevings en overlijdingskans passagiers per aantal ouders en kinderen aan boord",
            barmode="group",
            opacity=1.0
        )
        fig5.update_layout(xaxis_title="Aantal ouders en kinderen", yaxis_title="Percentage (%)")
        st.plotly_chart(fig5)

        df_famsize = bereken_percentages(df, 'Famsize')
        fig5 = px.bar(
            df_famsize,
            x="Famsize",
            y="Percentage",
            color="status",
            color_discrete_sequence=kleuren,
            category_orders={"status": ["Overleeft", "Overleden"]},
            
            barmode="group",
            opacity=1.0
        )
        fig5.update_layout(title = {'text':'Overlevings en overlijdingskans passagiers per aantal familieleden aan boord','x': 0.5, 'xanchor': 'center' },
                           xaxis_title="Aantal familieleden", yaxis_title="Percentage (%)")
        st.plotly_chart(fig5)       

    with tab2:
        st.title('2. Visualisaties')
        st.header('2.2 Andere visualisaties')
        st.write('Op deze pagina staan een aantal andere visualisaties.')
        st.subheader('2.2.1 Scatterplot')
        st.markdown(
        """De eerste visualisatie is een scatterplot met de variabelen fare, naamlengte en leeftijd.  
        In het dropdown menus zijn te kiezen welke variabelen je tegenover elkaar wilt zetten.  """)
        kolommen = ['Fare', 'Age', 'Nlength']

        xk = st.selectbox("Kies de variabele op de x-as:", kolommen, index=1)
        yk = st.selectbox("Kies de variabele op de y-as:", kolommen, index=0)

        # Log-schaal instellen afhankelijk van variabele
        logx = (xk == 'Fare')
        logy = (yk == 'Fare')

        # Handmatige kleurenmapping

        # Maak de scatterplot
        fig = px.scatter(
            df,
            x=xk,
            y=yk,
            color='status',
            color_discrete_map={'Overleeft': '#32CD32', 'Overleden': '#FF0000'},  # 👈 kleur op basis van status
            log_x=logx,
            log_y=logy
        )

        # Titel en assenlabels
        fig.update_layout(
            title={
                'text': f'Scatterplot {xk} vs {yk}',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=xk,
            yaxis_title=yk
        )

        # Pas log-as ticks aan indien nodig
        if logx:
            fig.update_xaxes(tickvals=[1, 10, 100, 1000], ticktext=['1', '10', '100', '1000'])
        if logy:
            fig.update_yaxes(tickvals=[1, 10, 100, 1000], ticktext=['1', '10', '100', '1000'])

        # Toon grafiek
        st.plotly_chart(fig, use_container_width=True)
        
    
        st.subheader('2.2.2 Afhankelijke verdelingen')
        st.markdown(
        """Verdeling van een gekozen variabele afhankelijk van ander gekozen variabele.  
        In de bovenste dropdownmenus kan links de variabele waarop gefilterd word gekozen en rechts op welke waarde.  
        In het dropdownmenu eronder kan de variabele gekozen worden waarvan de verdelin laten zien word  """)           

        c = ['status', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age', 'Famsize']

        col1, col2 = st.columns(2)

        with col1:
            filter = st.selectbox(
            'Kies de variabele om op te filteren',
            ['Gender', 'Klasse', 'Opstaplocatie', 'Status'],
            key='filter_selectbox'
            )

        # Maak een kopie van de kolommenlijst, zodat de originele behouden blijft
        kolommen = c.copy()

        if filter == 'Gender':
            kolommen.remove('Sex')
            with col2:
                gender = st.selectbox(
                'Kies het gender om op te filteren',
                df['Sex'].unique(),
                key='gender_selectbox'
                )
            dff = df[df['Sex'] == gender]

        elif filter == 'Klasse':
            kolommen.remove('Pclass')
            with col2:
               klasse = st.selectbox(
                'Kies de klasse om op te filteren',
                df['Pclass'].unique(),
                key='klasse_selectbox'
                )
            dff = df[df['Pclass'] == klasse]

        elif filter == 'Opstaplocatie':
            kolommen.remove('Embarked')
            embarked_namen = {
            'C': 'Cherbourg',
            'Q': 'Queenstown',
            'S': 'Southampton'
            }
            embarked_fullnames = [embarked_namen.get(code, code) for code in df['Embarked'].dropna().unique()]
            with col2:
                plek = st.selectbox(
                'Kies de opstaplocatie om op te filteren',
                embarked_fullnames,
                key='embarked_selectbox'
                )
            
            plek = [code for code, naam in embarked_namen.items() if naam == plek][0]


            dff = df[df['Embarked'] == plek]

        else:  # Status
            kolommen.remove('status')
            with col2:
                status = st.selectbox(
                'Kies de status om op te filteren',
                df['status'].unique(),
                key='status_selectbox'
                )
            dff = df[df['status'] == status]

        # Geef de gefilterde dataframe en aangepaste kolommenlijst door
        plot_histogram(dff, kolommen)

    with tab3:
        st.title('2. Visualisaties')
        st.header('2.3 Kaart Titanic')
        st.markdown("""In deze kaart wordt de route van de titanic laten zien met een paar toevoegingen:  
        - **Rode stippellijn**: Geplande route (tussen New York en ijsberg)  
        - **Zwarte stippellijn**: Vervoersroute slachtoffers.  
        - **Piratenschip**: Vervoerboot voor slachtoffers.  
        - **Reddingsboot**: Vervoer voor overlevenden.  
        - Verder zijn de **havens** en belangrijke punten gemarkeerd.""")
        m = folium.Map(location = [45.7833, -41.9167],zoom_start=4)  
        markers = [
            ([41.7267, -49.9483], '''<b>Schipwrak</b><br>
            Aantal slachtoffers: 424 ''' ,'shipwreck.jpg'),
            ([41.7833, -49.9167], '''<b>Ijsberg</b>''','iceberg.jpg'),
            ([50.8965, -1.3968], '''<b>Southampton</b><br>
            Aantal bijgekomen passagiers: 646<br>
            Gemiddelde fare: $27.08<br>
            Meest voorkomende klasse: 3<br>
            Gemiddelde leeftijd: 29.2 jaar''', 'port.jpg'),
            ([49.6469, -1.6222], '''<b>Cherbourg</b><br>
            Aantal bijgekomen passagiers: 168<br>
            Gemiddelde fare: $59.95<br>
            Meest voorkomende klasse: 1<br>
            Gemiddelde leeftijd: 30.2 jaar''', 'port.jpg'),
            ([51.85, -8.30], '''<b>Queenstown</b><br>
            Aantal bijgekomen passagiers: 77<br>
            Gemiddelde fare: $13.28<br>
            Meest voorkomende klasse: 3<br>
            Gemiddelde leeftijd: 28 jaar''', 'port.jpg'),
            ([40.6677, -74.0407], '''<b>New York</b><br>
            Geplande aankomstlocatie Titanic<br>
            Aankomstlocatie overlevenden''', 'port.jpg'),
            ([41.1972, -61.9945], '''<b>Reddingsboot</b><br>
            Overlevenden gered: 467<br>
            Gebracht naar: New York
            ''', 'lifeboat.jpg'),
            ([44.65, -63.57],'''<b>Halifax</b><br>
            Aankomstlocatie slachtoffers''', 'port.jpg'),
            ([43.1884,-56.7592],'''<b>Vervoer slachtoffers</b><br>
            Aantal slachtoffers: 424''', 'ship.jpg' ),
            ([46.8167,-29.1084], '''<b>Titanic</b><br>
            Aantal passagiers: 889<br>
            Gemiddelde fare: 32.2<br>
            Meest voorkomende klasse: 3<br>
            Gemiddelde leeftijd: 29.4''', 'pngimg.com - titanic_PNG31.jpg')
        ]
        v_crash = [[50.8965, -1.3968], [49.6469, -1.6222], [51.85, -8.30], [41.7833, -49.9167]  ]
        n_crash = [[41.7833, -49.9167],[40.6677, -74.0407]]
        o_trip = [[44.65, -63.57], [41.7267, -49.9483]]


        for latlon, popup, icon_path in markers:
            if icon_path: 
                if latlon == [41.7833, -49.9167]:
                    icon = folium.CustomIcon(icon_image=icon_path, icon_size=(50,50), icon_anchor=(40,40))
                else:
                    icon = folium.CustomIcon(icon_image=icon_path, icon_size=(30,30), icon_anchor=(20,20))
            else:
                icon = None
            folium.Marker(location=latlon, popup=folium.Popup(popup, max_width = 300), icon=icon).add_to(m)

        folium.PolyLine(
            locations=v_crash,
            color='green',       
            weight=3,           
            opacity=0.7,        
        ).add_to(m)
        folium.PolyLine(
            locations = n_crash,
            color='red',       
            weight=3,           
            opacity=0.7,        
            dash_array='5,10'   
        ).add_to(m)
        folium.PolyLine(
            locations = o_trip,
            color='black',       
            weight=3,           
            opacity=0.7,        
            dash_array='5,10'   
        ).add_to(m)


        st.set_page_config(layout="wide")
        st_folium(m, width = 1000)

elif keuze == 'Voorspelling':
    st.set_page_config(layout="wide")
    st.title('3. Voorspelling')
    st.header('3.1 Correlaties')
    corr = df.drop(columns=["PassengerId"]).corr(numeric_only=True)

   
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlatie Matrix", fontsize=14)

    
    st.pyplot(fig)
    plt.close(fig)

    st.header('3.2 Modellen')
    st.write('We hebben 2 modellen geprobeerd. Van welk model je de uitslagen wilt bekijken is in het dropdown menu te kiezen.')
    keuze = st.selectbox('Kies de modelvariant: ', ['Random forest','Logistische regressie'])
    if keuze == 'Random forest':
        dfr = df.copy()
# === ⿢ Selecteer nuttige kolommen ===
        features = ["Pclass", "Sex", "Age", "Embarked", "Nlength", "Famsize"]
        X = dfr[features].copy()
        y = dfr["Survived"]

        # === ⿣ Voorbewerking ===
        # Encodeer tekstvariabelen
        for col in ["Sex", "Embarked"]:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # === ⿤ Train/test-split ===
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # === ⿥ Model trainen ===
        rf = RandomForestClassifier(
            n_estimators=200,       # aantal bomen
            max_depth=6,            # beperkt boomdiepte om overfitting te voorkomen
            random_state=42
        )
        rf.fit(X_train, y_train)

        # === ⿦ Evalueren ===
        y_pred = rf.predict(X_val)
        # === Resultaten berekenen ===
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=["Overleden", "Overleefd"], output_dict=True)
        cm = confusion_matrix(y_val, y_pred)

        # === Titel ===
        st.header("Modelresultaten Titanic Random Forest")
        st.markdown("Hieronder zie je de prestaties van het model op de validatieset.")

        # === 1️⃣ Accuracy tonen ===
        st.metric(label="Accuracy", value=f"{acc:.3f}")

        # === 2️⃣ Confusiematrix ===
        st.subheader("Confusiematrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Overleden", "Overleefd"], yticklabels=["Overleden", "Overleefd"])
        ax.set_xlabel("Voorspeld")
        ax.set_ylabel("Werkelijk")
        ax.set_title("Confusiematrix (aantal passagiers)")
        st.pyplot(fig)

        # === 3️⃣ Precision / Recall / F1 per klasse ===
        st.subheader("Precision, Recall en F1-score per klasse")

        # Maak dataframe van het rapport
        report_df = pd.DataFrame(report).transpose().iloc[:2, :3]  # Alleen de 2 klassen
        report_df = report_df.reset_index().rename(columns={"index": "Klasse"})
        report_df["Klasse"] = ["Overleden", "Overleefd"]

        # Plotly bar chart
        fig_bar = px.bar(
            report_df.melt(id_vars="Klasse", var_name="Metric", value_name="Score"),
            x="Klasse", y="Score", color="Metric",
            barmode="group", text_auto=".2f",
            color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#1A535C"]  # rood, turquoise, donkerblauw
        )
        fig_bar.update_layout(title="Modelprestatie per klasse", yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

        # === 4️⃣ Tekstuele interpretatie ===
        st.subheader("Interpretatie")
        st.markdown(
        f"""
        - **Accuracy:** {acc:.2%} van de voorspellingen zijn correct.  
        - Het model is beter in het herkennen van **overleden passagiers (Recall = {report['Overleden']['recall']:.2f})**
        dan van **overlevenden (Recall = {report['Overleefd']['recall']:.2f})**.  
        - Precision laat zien dat wanneer het model voorspelt dat iemand **overleefde**, het daar meestal gelijk in heeft.  
        """
        )
        st.markdown("""De score op de gemaakte test-set is redelijk goed met 0.838, echter is de score op kaggle veel slechter met een score later dan 0.3.  
                    Dit kan komen door overfitting op de train data, maar de score van 0.838 zegt daar iets anders over. """)
        
    else:
        # Kies nuttige features
        features = ["Pclass", "Sex", "Age", "Embarked", 'Famsize', 'Nlength']
        df1 = df.copy()
        X = df1[features].copy()
        y = df1["Survived"]

        # Encodeer tekstvariabelen
        for col in ["Sex", "Embarked"]:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))


        # Train/test-split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model trainen
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Voorspellen
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=["Overleden", "Overleefd"], output_dict=True)
        cm = confusion_matrix(y_val, y_pred)

        # === Titel ===
        st.header("Resultaten Logistische Regressie")
        st.markdown("Hieronder zie je hoe goed het model de overlevingskansen voorspelt.")

        # === 1️⃣ Accuracy ===
        st.metric(label="Accuracy", value=f"{acc:.3f}")

        # === 2️⃣ Confusiematrix ===
        st.subheader("Confusiematrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Overleden", "Overleefd"], yticklabels=["Overleden", "Overleefd"])
        ax.set_xlabel("Voorspeld")
        ax.set_ylabel("Werkelijk")
        ax.set_title("Confusiematrix (aantal passagiers)")
        st.pyplot(fig)

        # === 3️⃣ Precision / Recall / F1 ===
        st.subheader("Precision, Recall en F1-score per klasse")

        report_df = pd.DataFrame(report).transpose().iloc[:2, :3]
        report_df = report_df.reset_index().rename(columns={"index": "Klasse"})
        report_df["Klasse"] = ["Overleden", "Overleefd"]

        fig_bar = px.bar(
            report_df.melt(id_vars="Klasse", var_name="Metric", value_name="Score"),
            x="Klasse", y="Score", color="Metric",
            barmode="group", text_auto=".2f",
            color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#1A535C"]
        )
        fig_bar.update_layout(title="Modelprestatie per klasse", yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

        # === 4️⃣ Coëfficiënten visualiseren ===
        st.subheader("Belangrijkheid van variabelen (coëfficiënten)")

        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        }).sort_values("Coefficient", ascending=False)

        fig_coef = px.bar(
            coef_df,
            x="Feature",
            y="Coefficient",
            color="Coefficient",
            color_continuous_scale="RdYlGn",
            title="Invloed van elke variabele op de overlevingskans"
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        # === 5️⃣ Interpretatie ===
        st.subheader("Interpretatie")
        st.markdown(
        f"""
        - **Accuracy:** {acc:.2%} van de voorspellingen zijn correct.  
        - De logistische regressie is een **lineair model**, wat betekent dat het probeert een rechte grens te trekken tussen overleefde en overleden passagiers.  
        - **Positieve coëfficiënten** verhogen de kans op overleving (bijv. vrouw, hogere klasse),  
        terwijl **negatieve coëfficiënten** de kans op overlijden verhogen (bijv. man, lagere klasse).  
        - **Recall en precision** helpen te begrijpen of het model meer fouten maakt bij overlevenden of overleden passagiers.  
        """
        )

        st.write("""De score van de gemaakte test-set is redelijk goed met een 0.821, Echter de score in kaggle is slechter met 0.656.  
                 Dit kan komen door overfitting op de train set, maar de score van 0.821 op de test set zegt daar iets anders over.""")