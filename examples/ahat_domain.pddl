; 为了解决partnr 的clean任务，修改 clean，不需要手里拿东西

(define (domain habitat_world_domain_final)
  (:requirements :strips :typing :equality 
    :disjunctive-preconditions
    :existential-preconditions
    :conditional-effects
  )


  (:types
    item
    agent - item ; 智能体
    location - item
      person - location ; 人类
      physobj - location              ; 所有物理对象的总父类
        room - physobj
        furniture - physobj ; 家具 (桌子、椅子等)
        device - physobj    ; 设备 (冰箱、微波炉等)
        object - physobj    ; 可拿取的小物品 (苹果、书等)
          container - object; 可拿取的小容器 (盒子、碗等)
    )


  (:predicates

    ; 人类相关状态
    (person_at ?p - person ?loc - physobj)
    (p_holding ?p - person ?i - physobj)

    ; -- 智能体状态 --
    (agent_at ?a - agent ?loc - location)
    (holding ?a - agent ?i - physobj) 
    (handempty ?a - agent)

    ; -- 位置与空间关系 --
    (item_on_surface ?i - physobj ?s - physobj)
    (item_in_receptacle ?i - physobj ?r - physobj)
    (furniture_in_room ?f - furniture ?r - room)
    (device_in_room ?d - device ?r - room)
    (next_to ?i1 - physobj ?i2 - physobj ?s - physobj)
    (can_only_place_one_object ?o - physobj)
    (can_be_moved ?o - physobj) ; 描述一个物理对象是否可以被移动（如家具、设备等）

    ; -- 属性与状态 --
    (can_be_opened ?c - physobj)
    (is_open ?c - physobj)
    (is_cleaning_tool ?o - physobj)
    (is_clean ?p - physobj)
    (require_floor_cleaner ?r - room) ; 描述一个房间是否需要地板清洁器来清洁（如地毯、地板等）
    (is_floor_cleaner ?f - physobj) ; 描述一个物理对象是否是地板清洁器（吸尘器等）

    (has_faucet ?f - physobj)
    (is_filled ?p - physobj)
    (can_dispense ?d - physobj) ; 描述一个设备是否具有分配/倒出液体的功能（例如咖啡机、饮水机）

    (is_powerable ?p - physobj)
    (is_powered_on ?p - physobj)
      
    ; heating相关谓词
    (is_heating_device ?p - physobj) ; 描述一个物理对象是否是加热设备（如微波炉、烤箱等）     

    ; 食物/物体处理状态
    (cooked ?o - physobj)
    (sliced ?o - physobj)
    (soaked ?o - physobj)

    ; 工具属性
    (is_slicer ?o - physobj) ; 可用于切片的工具（刀/切片器等）

    ; 烹饪相关属性：用于限制哪些物体可以作为“加热用容器/承载物”
    (is_heating_container ?o - physobj) ; 例如：pan/pot/baking_tray 等（不引入新 type）
  )

  ; =================================================================
  ; ACTIONS - 完整、逻辑严密的动作定义
  ; =================================================================

  ; Action: navigate
  (:action navigate
    :parameters (?ag - agent ?from - location ?to - location)
    :precondition (and (agent_at ?ag ?from) (not (= ?from ?to)))
    :effect (and (agent_at ?ag ?to) (not (agent_at ?ag ?from)))
  )

  ; Action: pick_from_surface
  (:action pick_from_surface
    :parameters (?ag - agent ?obj - physobj ?surface - physobj)
    :precondition (and 
      (agent_at ?ag ?obj) 
      (item_on_surface ?obj ?surface) 
      (handempty ?ag)
      (not (exists (?other_obj - physobj) (item_on_surface ?other_obj ?obj)))
    )
    :effect (and 
      (not (item_on_surface ?obj ?surface)) 
      (holding ?ag ?obj) 
      (not (handempty ?ag))
    )
  )
  
  ; Action: place_on_surface
  (:action place_on_surface
    :parameters (?ag - agent ?obj - physobj ?surface - physobj)
    :precondition (and 
        (agent_at ?ag ?surface) 
        (holding ?ag ?obj) 
        (not (= ?obj ?surface))     
        (not (item_on_surface ?surface ?obj))   
        (imply (can_only_place_one_object ?surface)
               (not (exists (?any_obj - physobj) (item_on_surface ?any_obj ?surface)))
        )
    )
    :effect (and 
        (item_on_surface ?obj ?surface) 
        (handempty ?ag) 
        (not (holding ?ag ?obj))
    )
)

  ; Action: pick_from_receptacle
  (:action pick_from_receptacle
    :parameters (?ag - agent ?obj - physobj ?r - physobj)
    :precondition (and (agent_at ?ag ?obj) (item_in_receptacle ?obj ?r)  (is_open ?r) (handempty ?ag))
    :effect (and (not (item_in_receptacle ?obj ?r)) (holding ?ag ?obj) (not (handempty ?ag)))
  )

  ; Action: place_in_receptacle
  (:action place_in_receptacle
    :parameters (?ag - agent ?obj - physobj ?r - physobj)
    :precondition (and (agent_at ?ag ?r) (holding ?ag ?obj)  (is_open ?r) (not (= ?obj ?r))
    )
    :effect (and (item_in_receptacle ?obj ?r) (handempty ?ag) )
  )
  
  ; Action: open
  (:action open
    :parameters (?ag - agent ?r - physobj)
    :precondition (and (agent_at ?ag ?r) (can_be_opened ?r) (not (is_open ?r)) (handempty ?ag))
    :effect (is_open ?r)
  )

  ; Action: close
  (:action close
    :parameters (?ag - agent ?r - physobj)
    :precondition (and (agent_at ?ag ?r)  (is_open ?r) (handempty ?ag))
    :effect (not (is_open ?r))
  )

  ; Action: power_on
  (:action power_on
    :parameters (?ag - agent ?p - physobj)
    :precondition (and
      (agent_at ?ag ?p)
      (is_powerable ?p)
      (not (is_powered_on ?p))
      (handempty ?ag)
      ; 若设备可打开，则必须处于关闭状态才能开机
      (imply (can_be_opened ?p) (not (is_open ?p)))
    )
    :effect (and (is_powered_on ?p))
  )

  ; Action: power_off
  (:action power_off
    :parameters (?ag - agent ?p - physobj)
    :precondition (and (agent_at ?ag ?p) (is_powerable ?p) (is_powered_on ?p) (handempty ?ag))
    :effect (not (is_powered_on ?p))
  )
  
  ; Action: fill_from_faucet
  (:action fill_from_faucet
    :parameters (?ag - agent ?c - physobj ?faucet_loc - physobj)
    :precondition (and (agent_at ?ag ?faucet_loc) (holding ?ag ?c) (has_faucet ?faucet_loc) (not (is_filled ?c)))
    :effect (is_filled ?c)
  )

  ; =================================================================
  ; !! 新增动作 !!
  ; 描述: 从一个具有分配功能的设备（如咖啡机）中，将液体装入一个手持容器。
  ; 这是解决“倒咖啡”问题的关键动作。
  ; =================================================================
  (:action dispense_from_device
    :parameters (?ag - agent ?d - physobj ?c - physobj)
    :precondition (and
        (agent_at ?ag ?d)      ; 1. 智能体必须在设备旁边
        (item_on_surface ?c ?d)       ; 2. 目标容器必须放在设备上
        (can_dispense ?d)      ; 3. 该设备必须具有分配功能
        (is_powered_on ?d)     ; 4. 设备电源已打开
        (not (is_filled ?c))   ; 5. 目标容器当前是空的
    )
    :effect (is_filled ?c)     ; 效果: 目标容器被填满
  )

  ; Action: pour_into
  (:action pour_into
    :parameters (?ag - agent ?c - physobj ?target - physobj)
    :precondition (and (holding ?ag ?c) (is_filled ?c) (agent_at ?ag ?target) 
      (not (is_filled ?target)))
    :effect (and 
              (not (is_filled ?c)) 
              (is_filled ?target)
            )
  )

  ; Action: cook_on_heating_device
  ; 描述: 将放在加热设备上的物体“烹饪/加热完成”
  (:action cook_on_heating_device
    ; 注意：不引入新的类别/type，这里的“盛放容器”仅用现有关系谓词 item_in_receptacle 来表达
    :parameters (?ag - agent ?o - physobj ?container - physobj ?d - physobj)
    :precondition (and
      (agent_at ?ag ?d)
      (or
        ; 若设备可打开，则需要将物体放入设备内部，并且设备处于打开状态
        (and (can_be_opened ?d) (is_open ?d) (item_in_receptacle ?o ?d))
        ; 若设备不可打开（如电炉台），则物体放在设备表面即可
        ; 约束：食物需要先放入一个容器中，并将容器放到加热设备上（例如：stove + 任意 cookware/container）
        (and
          (not (can_be_opened ?d))
          (is_heating_container ?container)
          (item_on_surface ?o ?container)
          (item_on_surface ?container ?d)
          (not (= ?o ?container))
          (not (= ?container ?d))
        )
      )
      (is_heating_device ?d)
      (is_powered_on ?d)
      (not (cooked ?o))
    )
    :effect (cooked ?o)
  )

  ; Action: slice_on_surface
  ; 描述: 将台面(或任意surface)上的物体切片（需要手持切片工具）
  (:action slice_on_surface
    :parameters (?ag - agent ?o - physobj ?surface - physobj ?slicer - physobj)
    :precondition (and
      (agent_at ?ag ?surface)
      (item_on_surface ?o ?surface)
      (holding ?ag ?slicer)
      (is_slicer ?slicer)
      (not (= ?o ?slicer))
      (not (sliced ?o))
    )
    :effect (sliced ?o)
  )

  ; Action: soak_with_faucet
  ; 描述: 在有水龙头的位置将手持物体浸泡/冲洗至“soaked”状态
  (:action soak_with_faucet
    :parameters (?ag - agent ?o - physobj ?faucet_loc - physobj)
    :precondition (and
      (agent_at ?ag ?faucet_loc)
      (holding ?ag ?o)
      (has_faucet ?faucet_loc)
      (not (soaked ?o))
    )
    :effect (soaked ?o)
  )
  
  ; ... 其他动作保持不变 ...
  (:action place_next_to_surface
    :parameters (?ag - agent ?obj_to_place - physobj ?obj_reference - physobj ?surface - furniture)
    :precondition (and
      (agent_at ?ag ?surface)
      (holding ?ag ?obj_to_place)
      (item_on_surface ?obj_reference ?surface)
      (not (= ?obj_to_place ?obj_reference))
      (not (= ?obj_to_place ?surface))
      (not (= ?obj_reference ?surface))
    )
    :effect (and
      (item_on_surface ?obj_to_place ?surface)
      (next_to ?obj_to_place ?obj_reference ?surface)
      (next_to ?obj_reference ?obj_to_place ?surface)
      (handempty ?ag)
      (not (holding ?ag ?obj_to_place))
    )
  )
  (:action place_next_to_inside
    :parameters (?ag - agent ?obj_to_place - physobj ?obj_reference - physobj ?receptacle - physobj)
    :precondition (and
      (agent_at ?ag ?receptacle)
      (holding ?ag ?obj_to_place)
      (item_in_receptacle ?obj_reference ?receptacle)
      (is_open ?receptacle)
      (not (= ?obj_to_place ?obj_reference))
      (not (= ?obj_to_place ?receptacle))
      (not (= ?obj_reference ?receptacle))
    )
    :effect (and
      (item_in_receptacle ?obj_to_place ?receptacle)
      (next_to ?obj_to_place ?obj_reference ?receptacle)
      (next_to ?obj_reference ?obj_to_place ?receptacle)
      (handempty ?ag)
      (not (holding ?ag ?obj_to_place))
    )
  )
  ;(:action clean_item
  ;  :parameters (?ag - agent ?obj - physobj ?target - object)
  ;  :precondition (and (holding ?ag ?obj) (not (is_clean ?target)) (is_cleaning_tool ?obj) (not (= ?obj ?target))
  ;  )
  ;  :effect (is_clean ?target)
  ;)
  ;(:action clean_furniture
  ;  :parameters (?ag - agent ?obj - physobj ?f - furniture )
  ;  :precondition (and (agent_at ?ag ?f) (not (is_clean ?f)) (holding ?ag ?obj) (not (= ?obj ?f))
  ;    (is_cleaning_tool ?obj)
  ;  )
  ;  :effect (is_clean ?f)
  ;)
  (:action clean
    :parameters (?ag - agent ?obj - physobj)
    :precondition (and (agent_at ?ag ?obj) (not (is_clean ?obj))
    )
    :effect (is_clean ?obj)
  )
  (:action clean_room_floor
    :parameters (?ag - agent ?obj - physobj ?r - room)
    :precondition (and (agent_at ?ag ?r) (not (is_clean ?r)) (holding ?ag ?obj) (not (= ?obj ?r))
      (is_floor_cleaner ?obj) (require_floor_cleaner ?r))
    :effect (is_clean ?r)
  )
  ; (:action set
  ;   :parameters (?ag - agent ?d - physobj)
  ;   :precondition (and (agent_at ?ag ?d) (not (is_setup ?d)) (handempty ?ag))
  ;   :effect (is_setup ?d)
  ; )
  (:action fill_device_with_water
    :parameters (?ag - agent ?d - physobj)
    :precondition (and (agent_at ?ag ?d) (not (is_filled ?d)) (has_faucet ?d))
    :effect (is_filled ?d)
  )
  (:action pour
    :parameters (?ag - agent ?c - physobj ?furn_with_faucet - furniture)
    :precondition (and (holding ?ag ?c) (is_filled ?c) (agent_at ?ag ?furn_with_faucet) 
      (has_faucet ?furn_with_faucet))
    :effect (and 
              (not (is_filled ?c))
            )
  )
  (:action move_furniture_to_room
    :parameters (?ag - agent ?f - furniture ?from - room ?to - room)
    :precondition (and 
        (can_be_moved ?f)
        (agent_at ?ag ?f) 
        (furniture_in_room ?f ?from) 
        (not (= ?from ?to)) 
        (handempty ?ag)
        (not (exists (?other_obj - physobj) (item_on_surface ?other_obj ?f)))
        (not (exists (?other_obj - physobj) (item_in_receptacle ?other_obj ?f)))
    )
    
    :effect (and 
        (not (furniture_in_room ?f ?from)) 
        (furniture_in_room ?f ?to) 
        (agent_at ?ag ?to) 
    )
)
  (:action move_device_to_room
    :parameters (?ag - agent ?d - device ?from - room ?to - room)
    :precondition (and (can_be_moved ?d) (agent_at ?ag ?d) (device_in_room ?d ?from) (not (= ?from ?to)) (handempty ?ag))
    :effect (and (not (device_in_room ?d ?from)) (device_in_room ?d ?to) (agent_at ?ag ?to))
  )
  (:action hand_over_to_person
    :parameters (?ag - agent ?p - person ?obj - physobj)
    :precondition (and (agent_at ?ag ?p) (holding ?ag ?obj)  (not (= ?ag ?p)))
    :effect (and 
      (p_holding ?p ?obj) ; 假设有一个谓词表示物品在人的手中
      (handempty ?ag)
    )
  )
)
