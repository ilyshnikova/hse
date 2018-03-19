import Data.Set (toList, fromList)
import Debug.Trace

uniquify lst = toList $ fromList lst

flat x = uniquify $ concat $ map (\(a,b) -> [a, b]) x

get_index components v = sum $ zipWith (\x i -> if v `elem` x then i else 0) components [1 .. ]

get_comp components v =
  let index = get_index components v
      with_first = take (index - 1) components
      with_last = drop (index) components
      comp = components !! (index - 1) in
    if index == 0 then
      (components, [])
    else
      (with_first ++ with_last, comp)

join :: [[Int]] -> (Int, Int) -> [[Int]]
join components (u, v) =
  let (without_u, with_u) = get_comp components u
      (without_u_v, with_v) = get_comp without_u v in
  without_u_v ++ [with_v ++ with_u]

connected_components edges = foldl join (map (\x -> [x]) (flat edges)) edges

main = do
  print $ connected_components [(1,2),(2,3),(1,3),(3,4),(4,2),(5,6), (7,8), (9,9)]
