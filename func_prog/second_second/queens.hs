queens_rec cur_states y =
  if y == 9 then
    [cur_states]
  else
    concat (map f [1 .. 8]) where
      f x =
        if
          y == 1 ||
          all (==True) (zipWith (\x' y' -> (x + y /= x' + y') && (x - y /= x' - y') && x /= x' && y /= y') cur_states [1 .. y])
        then
          queens_rec (cur_states ++ [x]) (y + 1)
        else
          [[]]


queens = filter (/=[]) $ queens_rec [] 1

main = do
  print $ queens
