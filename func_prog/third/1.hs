import Data.Bits (xor)

update states index count =
  take index states ++ [((states !! index) - count)] ++ drop (index + 1) states

next_step states =
  let xor_res = foldl xor 0 states
      diff = zip [0 ..] (map (\x -> x - xor x (if xor_res == 0 then x else xor_res)) states)
  in filter (\(i, x) -> x > 0 && x <= states !! i) diff !! 0

nim states = do
  print states
  if all (== 0) states then do
    putStrLn "computer win"
  else do
    index <- readLn :: IO Int
    count <- readLn :: IO Int
    if 0 <= index && index < length states && 0 < count && count <= states !! index then do
      let new_states = update states index count
      if all (== 0) new_states then do
        putStrLn "you win"
      else do
        let (next_index, next_count) = next_step new_states
        putStrLn ("heap index: " ++ show next_index)
        putStrLn ("stones number: " ++ show next_count)
        nim $ update new_states next_index next_count
    else do
      putStrLn  "error input"
      nim states

main = do
  nim [2, 2]
