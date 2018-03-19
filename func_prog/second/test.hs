flat x = concat $ map (\(a,b) -> [a, b]) x

my_max a = max (flat a)

main = do
  print $ take 5 [0,1,2,3,4,5,6,7]
  print $ drop 6 [0,1,2,3,4,5,6,7]
  print $ [0,1,2,3,4,5,6,7] !! 5

