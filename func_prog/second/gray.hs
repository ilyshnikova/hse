gray 0 = [[]]

gray n = map (False :) (gray (n - 1)) ++ reverse (map (True :) (gray (n - 1)))

hamming first second =  sum (zipWith f first second) where f x y = if x == y then 0 else 1

main = do
  print $ all (== 1) (zipWith hamming (gray 10) (tail (gray 10)))

