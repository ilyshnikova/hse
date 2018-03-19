paths_rec edges to cur_path cur_vertex =
  if cur_vertex == to then
    [cur_path]
  else
    concat $ map (
      \(f,t) ->
        if ((f /= cur_vertex) || (t `elem` cur_path)) then
          [[]]
        else
          paths_rec edges to (cur_path ++ [t]) t
    ) edges

paths from to edges = filter (/= []) $ paths_rec edges to [from] from

main = do
 print $ paths 1 4 [(1,2),(2,3),(1,3),(3,4),(4,2),(5,6)]
