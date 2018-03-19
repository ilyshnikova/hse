decsplit 0 = []
decsplit n = [n `mod` 10] ++ decsplit(n `div` 10)

luhn n = sum (zipWith f [0 .. ] (decsplit n)) `mod` 10 == 0  where
  f i x = sum $ decsplit $ x * ((i `mod` 2) + 1)

main = do
  print $ luhn 4561261212345464
  print $ luhn 4561261212345467
