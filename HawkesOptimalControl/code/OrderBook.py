import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from datetime import time as _time
from config import _ANCHOR_DATE

class OrderBook:
    def __init__(self, message_file, orderbook_file=None, ticker="AMZN"):
        """
        Gestisce il caricamento da dati LOBSTER (Message + Orderbook).
        
        Args:
            message_file (str): Percorso al file message_10.csv
            orderbook_file (str): Percorso al file orderbook_10.csv
            ticker (str): Simbolo del titolo (per i plot)
        """
        self.ticker = ticker
        
        # Gestione percorsi
        self.msg_path = Path(message_file)
        if orderbook_file:
            self.ob_path = Path(orderbook_file)
        else:
            # Prova a indovinare se non fornito (assumendo stessa cartella)
            self.ob_path = Path(str(message_file).replace("message", "orderbook"))

        print(f"Loading LOBSTER data from:\n MSG: {self.msg_path}\n LOB: {self.ob_path}...")
        
        # Caricamento e Processamento Dati
        self.LimitOrderBook = self._load_lobster_data()
        
        print("Data loaded successfully.")
        print(f"Time Range: {self.LimitOrderBook['time'].min():.1f}s - {self.LimitOrderBook['time'].max():.1f}s")

    def _load_lobster_data(self):
        """Legge i CSV Raw e crea il DataFrame strutturato per il progetto."""
        
        # 1. Load Message File
        # Columns: Time, Type, OrderID, Size, Price, Direction
        msg = pd.read_csv(self.msg_path, header=None, 
                          names=['time', 'type', 'id', 'size', 'price', 'dir'])
        
        # 2. Load Orderbook File
        # Columns: P1_Ask, V1_Ask, P1_Bid, V1_Bid, ... up to Level 10
        lob = pd.read_csv(self.ob_path, header=None)
        
        # 3. Costruzione DataFrame Principale
        df = pd.DataFrame()
        df['time'] = msg['time']
        df['ts'] = _ANCHOR_DATE + pd.to_timedelta(df['time'], unit='s')
        
        # --- Prezzi e Livelli (Normalizzazione LOBSTER: prezzi sono int * 10000) ---
        # --- Store raw message fields needed for calibration ---
        df["lob_action"] = msg["type"].astype(int)
        df["lob_dir"]    = msg["dir"].astype(int)      # LOBSTER convention (side of resting LO)
        df["lob_size"]   = msg["size"].astype(float)   # <-- IMPORTANT for A,k calibration
        df["lob_price"]  = msg["price"].astype(float) / 10000.0

        # --- Full depth L1..L10 (prices normalized; volumes as floats) ---
        MAX_L = 10
        for L in range(1, MAX_L + 1):
            base = (L - 1) * 4
            df[f"ask_price_{L}"] = lob.iloc[:, base + 0].astype(float) / 10000.0
            df[f"ask_size_{L}"]  = lob.iloc[:, base + 1].astype(float)
            df[f"bid_price_{L}"] = lob.iloc[:, base + 2].astype(float) / 10000.0
            df[f"bid_size_{L}"]  = lob.iloc[:, base + 3].astype(float)

        # Legacy L1 aliases (so you don't break old code)
        df["ask_price"] = df["ask_price_1"]
        df["ask_size"]  = df["ask_size_1"]
        df["bid_price"] = df["bid_price_1"]
        df["bid_size"]  = df["bid_size_1"]

        df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2.0

        # Execution price (legacy)
        df["execution"] = df["lob_price"]

        # Bid/Ask legacy columns (older code uses 'bid'/'ask')
        df["bid"] = df["bid_price"]
        df["ask"] = df["ask_price"]

        
        # --- CALCOLO IMBALANCE PROFONDA (L1-L5) ---
        # Pesi per l'imbalance pesata (WOBI)
        weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        
        # Estrazione volumi L1-L5 (indici: Ask=1,5,9,13,17 / Bid=3,7,11,15,19)
        ask_vols = lob.iloc[:, [1, 5, 9, 13, 17]].values
        bid_vols = lob.iloc[:, [3, 7, 11, 15, 19]].values
        
        w_ask = np.dot(ask_vols, weights)
        w_bid = np.dot(bid_vols, weights)
        
        # Salva rho_5L nel dataframe: (Bid - Ask) / (Bid + Ask)
        weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=float)

        ask_vols = np.column_stack([df[f"ask_size_{L}"].values for L in range(1, 6)])
        bid_vols = np.column_stack([df[f"bid_size_{L}"].values for L in range(1, 6)])

        w_ask = ask_vols @ weights
        w_bid = bid_vols @ weights

        df["rho_5L"] = (w_bid - w_ask) / (w_bid + w_ask + 1e-9)

        
        # --- Mapping Eventi LOBSTER -> Formato Utente ---
        # LOBSTER Types:
        # 1: Submission
        # 2: Cancellation (Partial)
        # 3: Deletion (Total)
        # 4: Execution (Visible)
        # 5: Execution (Hidden)
        # 7: Halt
        
        
        # Colonne "Human Readable" per compatibilità plot legacy
        type_map = {
            1: "Submission of a new limit order",
            2: "Deletion", # Mappiamo Cancel a Deletion per semplicità
            3: "Deletion",
            4: "Execution of a visible limit order",
            5: "Execution of a hidden limit order",
            7: "Trading Halt"
        }
        df['operation'] = msg['type'].map(type_map).fillna("Unknown")
        
        # Direction string (Legacy)
        # Nota: In LOBSTER, Dir=1 è Buy, -1 è Sell.
        dir_map = {1: "BUY limit order", -1: "SELL limit order"}
        df['direction'] = msg['dir'].map(dir_map).fillna("Unknown")
        
        # Execution Price (Normalizzato)
        df['execution'] = msg['price'] / 10000.0
        
        # Bid/Ask legacy columns (per compatibilità con codici vecchi che usano 'bid'/'ask')
        df['bid'] = df['bid_price']
        df['ask'] = df['ask_price']
        
        return df

    def to_hawkes_events_10d_robust(self, debug=False):
        """
        Estrae gli eventi Hawkes (L, C, E, M) direttamente dai codici LOBSTER.
        Molto più veloce e preciso del parsing stringhe.
        """
        df = self.LimitOrderBook
        
        # 1. Definizioni Base basate su Codici LOBSTER
        # Type 1 = Limit Order (L)
        is_L = (df['lob_action'] == 1)
        
        # Type 2 (Cancel) or 3 (Delete) = Cancellation (C)
        is_C = (df['lob_action'].isin([2, 3]))
        
        # Type 4 or 5 = Execution (E) -> Questo è un Market Order che colpisce il book
        is_E = (df['lob_action'].isin([4, 5]))
        
        # Market Order (M) è identico all'evento Esecuzione nella vista LOBSTER
        # (Un MO è ciò che causa una Execution)
        # Nel modello 4D/8D, E è l'evento "MO arriva e mangia liquidità"
        
        # Type 1 con prezzo diverso da best? (Opzionale per HE/D refinement)
        # Per ora usiamo logica standard base
        
        # 2. Direzione
        # LOBSTER: 'direction' indica il lato dell'ORDINE LIMITE a cui l'evento si riferisce
        # (1 = buy limit order sul bid, -1 = sell limit order sull'ask).
        # Questo è cruciale per le esecuzioni:
        #   - Se viene eseguito un BUY limit (dir=+1), l'aggressore è un market SELL.
        #   - Se viene eseguito un SELL limit (dir=-1), l'aggressore è un market BUY.
        # Nel modello della tesi vogliamo eventi aggressivi (market orders):
        #   E_a = market BUY che colpisce l'ask (=> esegue un sell limit, dir=-1)
        #   E_b = market SELL che colpisce il bid (=> esegue un buy  limit, dir=+1)
        
        is_buy = (df['lob_dir'] == 1)
        is_sell = (df['lob_dir'] == -1)

        # 3. Costruzione Dizionario Eventi
        # Nota: Qui mappiamo sugli eventi Hawkes 8D (o 10D esteso)
        # L=Limit, C=Cancel, E=Execution (Market Order), D=Delete (Spesso C e D accorpati)
        
        # Componenti standard (L, C, E)
        # Bid Side
        L_b = df[is_L & is_buy]['time'].values
        C_b = df[is_C & is_buy]['time'].values
        # ATTENZIONE: per le esecuzioni, il segno e' quello del limite eseguito (non dell'aggressore)
        E_b = df[is_E & is_buy]['time'].values  # market SELL (aggressore) -> esegue buy limit (dir=+1)
        
        # Ask Side
        L_a = df[is_L & is_sell]['time'].values
        C_a = df[is_C & is_sell]['time'].values
        E_a = df[is_E & is_sell]['time'].values # market BUY  (aggressore) -> esegue sell limit (dir=-1)
        
        # Mapping per config.py COMPONENTS
        # User config: L_b, E_b, HE_b, D_b ...
        # Semplificazione: Mappiamo tutto ciò che non è "Limit at Touch" in "Limit" generico per ora
        # O se vuoi distinzione, serve logica prezzi.
        
        # Per semplicità e robustezza:
        events = {
            "L_b": np.sort(L_b),
            "C_b": np.sort(C_b), # Usiamo C_b per Cancel/Delete
            "E_b": np.sort(E_b), # Market Sell (colpisce bid)
            "D_b": np.sort(C_b), # Alias se il config lo richiede
            
            "L_a": np.sort(L_a),
            "C_a": np.sort(C_a),
            "E_a": np.sort(E_a), # Market Buy (colpisce ask)
            "D_a": np.sort(C_a)
        }
        
        # HE (Hidden Exec) - LOBSTER Type 5
        is_HE = (df['lob_action'] == 5)
        # Hidden executions: stessa convenzione di direction
        events["HE_b"] = np.sort(df[is_HE & is_buy]['time'].values)   # market SELL aggressore
        events["HE_a"] = np.sort(df[is_HE & is_sell]['time'].values)  # market BUY aggressore

        # Durata totale dataset (per normalizzazioni)
        T_day = df['time'].max() - df['time'].min()
        
        if debug:
            print(f"Extracted Events (LOBSTER Logic):")
            print(f" L_b: {len(events['L_b'])}, L_a: {len(events['L_a'])}")
            print(f" E_b (MO Sell): {len(events['E_b'])}, E_a (MO Buy): {len(events['E_a'])}")
            print(f" C_b: {len(events['C_b'])}, C_a: {len(events['C_a'])}")

        return events, T_day

    # --- Metodi Plot Legacy (Mantenuti per compatibilità) ---
    def plot_bid_ask_exec(self):
        df = self.LimitOrderBook
        plt.figure(figsize=(12, 6))
        plt.plot(df['ts'], df['ask_price'], label='Ask', color='red', linewidth=0.5, alpha=0.7)
        plt.plot(df['ts'], df['bid_price'], label='Bid', color='green', linewidth=0.5, alpha=0.7)
        
        # Plot executions
        execs = df[df['lob_action'].isin([4, 5])]
        plt.scatter(execs['ts'], execs['execution'], color='black', s=10, label='Executions', zorder=5)
        
        plt.title(f"{self.ticker} Limit Order Book & Executions")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def cut_and_plot(self, start_time, end_time):
        """Taglia e plotta una finestra temporale specifica."""
        # start_time/end_time sono oggetti datetime.time
        # Convertiamo in secondi dal midnight per filtrare
        t0 = start_time.hour*3600 + start_time.minute*60 + start_time.second
        t1 = end_time.hour*3600 + end_time.minute*60 + end_time.second
        
        mask = (self.LimitOrderBook['time'] >= t0) & (self.LimitOrderBook['time'] <= t1)
        sub_df = self.LimitOrderBook.loc[mask]
        
        if sub_df.empty:
            print("No data in selected range.")
            return

        plt.figure(figsize=(12, 6))
        plt.step(sub_df['ts'], sub_df['ask_price'], label='Ask', color='red', where='post')
        plt.step(sub_df['ts'], sub_df['bid_price'], label='Bid', color='green', where='post')
        
        execs = sub_df[sub_df['lob_action'].isin([4, 5])]
        plt.scatter(execs['ts'], execs['execution'], color='black', marker='x', s=50, label='Trade')
        
        plt.title(f"Zoom: {start_time} - {end_time}")
        plt.legend()
        plt.show()