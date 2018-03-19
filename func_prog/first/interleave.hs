interleave x ys = map f [0 .. length ys]  where
  f i = take i ys ++ [x] ++ drop i ys

main = do
  print $ interleave 5 [2,34,10]
