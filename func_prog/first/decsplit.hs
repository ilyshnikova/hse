decsplit 0 = []
decsplit n = [n `mod` 10] ++ decsplit(n `div` 10)

main = do
  print $ decsplit 112345
