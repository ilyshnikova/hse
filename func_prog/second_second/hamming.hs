hamming first second =  sum (zipWith f first second) where f x y = if x == y then 0 else 1

main = do
  print $ (hamming [1,2,3,4,5] [1,4,3,8,5])
