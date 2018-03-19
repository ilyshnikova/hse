fib = 1 : 1 : map f [2..] where
  f n = fib !! (n - 1) + fib !! (n - 2)

main = do
  print $ take 10 fib
