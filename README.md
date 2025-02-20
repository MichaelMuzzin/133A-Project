133A final project. 
PT 1  ( 02-21 )
  
  Pt A - Our data was Provided via Alpaca API at 1 minute intervals. It contains the historical price and volumetric data for SOL/USD (Solana in USD). It had to be periodically collected in batches to avoid the API limit rate. We fetch around 20,000 lines, but filter down to ensure at least 10,000 rows, currnetly targeting ~15,000 rows. The Columns that are extracted are a mixture of raw and calucalted data. Raw features include {TimeStamp, Symbol, Open, High, Low, Close, Volume}. Derived technical indicators make up the remainder of the 30 columns. 
  
  Pt B - Our classification target will be binary, indicating whether the price will increase or decrease in the next time stamp.

The current trajectory that the project will take is likely a cyrpto (solana) price predictoin bot. I believe that we will try to do this with a mixture of LSTM's and Random Forests for variety. 
