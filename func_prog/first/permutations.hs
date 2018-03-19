interleave x ys = map f [0 .. length ys]  where
  f i = take i ys ++ [x] ++ drop i ys

permutations [] = [[]]

permutations (x : ys) = concat $ map (interleave x) (permutations ys)

main = do
  print $ permutations [1,2,3]



