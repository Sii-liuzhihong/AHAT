; Hand-written smoke domain, not a copy of private training assets.
(define (domain habitat_world_domain_final)
 (:requirements :strips :typing)
 (:types location agent - object physobj - location room furniture - physobj)
 (:predicates (agent_at ?a - agent ?loc - location)
              (handempty ?a - agent)
              (furniture_in_room ?f - furniture ?r - room)
              (is_clean ?p - physobj))
 (:action wipe
  :parameters (?a - agent ?f - furniture ?r - room)
  :precondition (and (agent_at ?a ?r) (furniture_in_room ?f ?r))
  :effect (is_clean ?f)))
