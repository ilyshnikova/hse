fib = 1 : 1 : map f [2..] where
  f n = fib !! (n - 1) + fib !! (n - 2)

fibnya = map f fib where
  f n = concat $ take n $ repeat "nya"

main = do
  print $ take 10 fibnya
