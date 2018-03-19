a_b 'a' = 'b'
a_b 'b' = 'a'

thue 1 = "a"
thue i = thue (i - 1) ++ map a_b (thue (i - 1))


thue1 1 = "b"
thue1 i = thue1 (i - 1) ++ map a_b (thue1 (i - 1))

thue_infinite = "a" ++ concat (map thue1 [1 .. ])


main = do
  print $ thue 5
  print $ take 16 thue_infinite
