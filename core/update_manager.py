'''
Manages the automatic update process for FieldForge SDF systems.
Includes debouncing, throttling, triggering mesh regeneration,
and the Blender dependency graph handler.
This version uses multi-threading for mesh generation to keep the UI responsive.
'''

import bpy
import time
import math
from mathutils import Matrix, Vector
import threading
import queue

# Use relative imports assuming this file is in FieldForge/core/
from .. import constants
from .. import utils # For find_parent_bounds, get_all_bounds_objects, get_bounds_setting, find_result_object
from . import state # For get_current_sdf_state, has_state_changed
from . import sdf_logic # For process_sdf_hierarchy

# Import libfive if available (needed for run_sdf_update)
try:
    import libfive.stdlib as lf
    _lf_imported_ok = True
except ImportError:
    _lf_imported_ok = False
    # Define dummy lf object if needed for type hinting or basic structure,
    # but run_sdf_update will check _lf_imported_ok anyway.
    class LFDummy:
        def emptiness(self): return None # Simulate emptiness
    lf = LFDummy()


# --- Global State Dictionaries (Managed by this module) ---
# Keys are generally bounds_obj.name

# Flags indicating an update is scheduled or running for a specific bounds
_updates_pending = {}
# Caches the last known state dictionary used for a successful update
_sdf_update_caches = {}
# State for multithreading
_worker_threads = {}
_color_worker_threads = {}
# Stores the current div being displayed for each bounds object
_current_divs = {}
# Stores the target div for each bounds object (can be lower than current)
_target_divs = {}

MAX_DIV = 5 # Corresponds to lowest resolution (highest div)
MIN_DIV = 0 # Corresponds to highest resolution (lowest div)


def clear_link_caches(): # Call from clear_timers_and_state
    state.clear_link_caches()

# --- Cache Update ---

def update_sdf_cache(new_state: dict, bounds_name: str):
    """ Updates the cache for a specific bounds object with the new state. """
    global _sdf_update_caches
    if new_state and bounds_name:
        # State should already contain copies (e.g., matrix.copy())
        _sdf_update_caches[bounds_name] = new_state


# --- Debounce and Throttle Logic (Per Bounds) ---

def check_and_trigger_update(bounds_name: str, reason: str="unknown"):
    """
    Checks if an update is needed for a specific bounds hierarchy based on state change.
    If needed and auto-update is on, resets the debounce timer for viewport updates.
    """
    global _updates_pending, _sdf_update_caches # Access relevant state dicts

    is_viewport_update = True # This function is always called for auto-updates (viewport context)

    context = bpy.context
    if not context or not context.scene: return # Context/Scene might not be ready

    scene = context.scene
    bounds_obj = scene.objects.get(bounds_name)
    if not bounds_obj or not bounds_obj.get(constants.SDF_BOUNDS_MARKER):
        # Clean up potentially orphaned state if object is gone
        _updates_pending.pop(bounds_name, None)
        _sdf_update_caches.pop(bounds_name, None)
        _cancel_and_clear_worker(bounds_name)
        return

    # Check the auto-update setting ON THE BOUNDS OBJECT
    if not utils.get_bounds_setting(bounds_obj, "sdf_auto_update"):
        return # Auto update disabled for this system

    # Don't re-trigger if an update is already pending/running for this bounds
    if _updates_pending.get(bounds_name, False):
        return

    # Get the current state ONLY if necessary checks pass
    current_state = state.get_current_sdf_state(context, bounds_obj)
    if not current_state: # Handle case where state gathering fails
        print(f"FieldForge WARN (check_trigger): Could not get current state for {bounds_name}.")
        return

    # Compare current state to the cached state for this specific bounds
    cached_state = _sdf_update_caches.get(bounds_name)
    if state.has_state_changed(current_state, cached_state): # Pass cached state directly
        # State has changed, directly trigger update (no debounce/throttle)
        run_sdf_update(bounds_name, current_state, is_viewport_update=True)

# --- Core Update Function (Now split into Thread Starter and Result Applicator) ---

def _mesh_generation_worker(result_q, is_cancelled_flag, shape, bounds_name, base_resolution, current_div, xyz_min, xyz_max):
    """
    Worker function to be run in a separate thread.
    Performs the heavy mesh generation and puts the result in a queue.
    """
    if is_cancelled_flag.is_set():
        result_q.put(None) # Signal that we cancelled before starting
        return

    try:
        t_start_worker = time.perf_counter()
        # This is the slow, CPU-intensive part
        mesh_data = shape.get_mesh(xyz_min=xyz_min, xyz_max=xyz_max, resolution=max(3, int(base_resolution / (1 << current_div))))
        meshing_time = time.perf_counter() - t_start_worker

        # DEBUG: Log mesh generation results
        vert_count = len(mesh_data[0]) if mesh_data and mesh_data[0] else 0
        tri_count = len(mesh_data[1]) if mesh_data and mesh_data[1] else 0
        print(f"FieldForge DEBUG: Mesh generation for {bounds_name} at resolution {max(3, int(base_resolution / (1 << current_div)))} -> Verts: {vert_count}, Tris: {tri_count}, Time: {meshing_time:.4f}s")

        if is_cancelled_flag.is_set():
            result_q.put(None) # Cancelled during generation, discard result
            return
        result_q.put((mesh_data, meshing_time, current_div)) # Put the result, time, and div in the queue
    except Exception as e:
        print(f"FieldForge Thread ERROR: libfive mesh generation failed for {bounds_name}: {e}")
        result_q.put(Exception(f"Meshing failed: {e}")) # Put exception in queue to report it

def _color_calculation_worker(result_q, is_cancelled_flag, verts_copy, matrix_world_copy, individual_sdf_shapes, influence_threshold):
    if is_cancelled_flag.is_set():
        result_q.put(None)
        return

    try:
        print(f"FieldForge DEBUG: Color calculation worker started for {len(verts_copy)} vertices.")
        vertex_colors = {}
        for i in range(len(verts_copy)):
            if is_cancelled_flag.is_set():
                result_q.put(None)
                return

            vert_co = verts_copy[i]
            vertex_world_pos = matrix_world_copy @ vert_co
            
            total_weight = 0.0
            final_color = Vector((0.0, 0.0, 0.0, 0.0))

            for shape_name, (sdf_shape, color) in individual_sdf_shapes.items():
                if not (sdf_shape is None or sdf_shape is lf.emptiness()):
                    try:
                        dist = sdf_shape(vertex_world_pos.x, vertex_world_pos.y, vertex_world_pos.z)
                        if abs(dist) < influence_threshold:
                            weight = 1.0 - (abs(dist) / influence_threshold)
                            final_color += Vector(color) * weight
                            total_weight += weight
                    except Exception as e:
                        print(f"FieldForge DEBUG: Error during color calculation for shape {shape_name}: {e}")
            
            if total_weight > 0:
                final_color /= total_weight
            
            vertex_colors[i] = final_color

        print(f"FieldForge DEBUG: Color calculation worker finished. Found colors for {len(vertex_colors)} vertices.")
        result_q.put(vertex_colors)
    except Exception as e:
        result_q.put(e)

def _apply_color_data_from_worker(result_obj_name, expected_vert_count, bounds_name, update_id):
    global _color_worker_threads, _updates_pending
    if result_obj_name not in _color_worker_threads:
        return None

    thread, result_q, timer, is_cancelled_flag = _color_worker_threads[result_obj_name]

    try:
        result_data = result_q.get_nowait()
    except queue.Empty:
        return 0.1 # Data not ready, reschedule timer

    # --- Data is ready, now we can pop the worker and process the data ---
    _color_worker_threads.pop(result_obj_name)

    is_cancelled_flag.set()
    if timer and bpy.app.timers.is_registered(timer):
        bpy.app.timers.unregister(timer)

    if isinstance(result_data, Exception):
        print(f"FieldForge ERROR: Color worker failed for {result_obj_name}: {result_data}")
        _updates_pending[bounds_name] = False
        return None

    if result_data is None:
        _updates_pending[bounds_name] = False
        return None

    result_obj = bpy.data.objects.get(result_obj_name)
    if not result_obj or result_obj.type != 'MESH':
        _updates_pending[bounds_name] = False
        return None

    # --- ID Check ---
    mesh_update_id = result_obj.get(constants.SDF_UPDATE_ID_PROP)
    if mesh_update_id != update_id:
        print(f"FieldForge DEBUG: Stale color data for {result_obj_name}. Mesh ID: {mesh_update_id}, Worker ID: {update_id}")
        _updates_pending[bounds_name] = False
        return None # Stale data, abort

    print(f"FieldForge DEBUG: Applying color data to {result_obj_name}. Expected verts: {expected_vert_count}, actual verts: {len(result_obj.data.vertices)}, calculated colors: {len(result_data)}")

    if len(result_obj.data.vertices) != expected_vert_count:
        _updates_pending[bounds_name] = False
        return None

    color_attr_name = constants.SDF_COLOR_ATTRIBUTE_NAME
    if color_attr_name not in result_obj.data.color_attributes:
        result_obj.data.color_attributes.new(name=color_attr_name, type='FLOAT_COLOR', domain='POINT')
    
    color_layer = result_obj.data.color_attributes[color_attr_name]
    for vert_index, color in result_data.items():
        color_layer.data[vert_index].color = color

    print(f"FieldForge DEBUG: Applied color attribute data to {result_obj_name}")
    _updates_pending[bounds_name] = False
    return None

def _manage_sdf_color_attribute(result_obj: bpy.types.Object, bounds_obj: bpy.types.Object, context: bpy.types.Context, update_id: int):
    global _color_worker_threads
    if not result_obj or result_obj.type != 'MESH' or not _lf_imported_ok:
        return

    # Cancel any existing color worker for this object
    if result_obj.name in _color_worker_threads:
        _, _, timer, is_cancelled_flag = _color_worker_threads.pop(result_obj.name)
        is_cancelled_flag.set()
        if timer and bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)

    # Ensure the color attribute exists before starting the worker
    color_attr_name = constants.SDF_COLOR_ATTRIBUTE_NAME
    if color_attr_name not in result_obj.data.color_attributes:
        result_obj.data.color_attributes.new(name=color_attr_name, type='FLOAT_COLOR', domain='POINT')

    sdf_sources = []
    def find_sdf_sources_recursive(obj):
        if utils.is_sdf_source(obj) and obj.visible_get(view_layer=context.view_layer):
            sdf_sources.append(obj)
        for child in obj.children:
            find_sdf_sources_recursive(child)
    find_sdf_sources_recursive(bounds_obj)

    if not sdf_sources:
        return

    individual_sdf_shapes = {}
    for source_obj in sdf_sources:
        shape = sdf_logic.reconstruct_shape(source_obj)
        if not (shape is None or shape is lf.emptiness()):
            transformed_shape = sdf_logic.apply_blender_transform_to_sdf(
                shape, source_obj.matrix_world.inverted())
            individual_sdf_shapes[source_obj.name] = (transformed_shape, source_obj.sdf_color)

    if not result_obj.data.vertices:
        return

    influence_threshold = 0.1
    result_q = queue.Queue()
    is_cancelled_flag = threading.Event()

    verts_copy = [v.co.copy() for v in result_obj.data.vertices]
    matrix_world_copy = result_obj.matrix_world.copy()
    expected_vert_count = len(verts_copy)

    # If there are no vertices, no need to start the worker
    if expected_vert_count == 0:
        return

    worker_thread = threading.Thread(
        target=_color_calculation_worker,
        args=(result_q, is_cancelled_flag, verts_copy, matrix_world_copy, individual_sdf_shapes, influence_threshold)
    )

    result_timer = bpy.app.timers.register(
        lambda: _apply_color_data_from_worker(result_obj.name, expected_vert_count, bounds_obj.name, update_id),
        first_interval=0.1
    )

    _color_worker_threads[result_obj.name] = (worker_thread, result_q, result_timer, is_cancelled_flag)
    worker_thread.start()

def _ensure_sdf_material(result_obj: bpy.types.Object):
    mat_name = constants.SDF_MATERIAL_NAME
    material = bpy.data.materials.get(mat_name)

    if not material:
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        bsdf = nodes.get('Principled BSDF')
        if bsdf:
            nodes.remove(bsdf)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        
        attr_node = nodes.new(type='ShaderNodeAttribute')
        attr_node.attribute_name = constants.SDF_COLOR_ATTRIBUTE_NAME
        attr_node.location = (-200, 200)
        
        links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
        
        output_node = nodes.get('Material Output')
        if output_node:
            links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    if not result_obj.data.materials or result_obj.data.materials[0] != material:
        if not result_obj.data.materials:
            result_obj.data.materials.append(material)
        else:
            result_obj.data.materials[0] = material

def _apply_mesh_data_from_worker(bounds_name: str, trigger_state: dict, is_viewport_update: bool):
    """
    Timer callback for the main thread. Checks the result queue from the worker.
    If a result is available, it applies the new mesh data to the Blender object.
    """
    global _worker_threads, _updates_pending, _sdf_update_caches, _current_divs, _target_divs
    
    if bounds_name not in _worker_threads:
        return None # Worker was cancelled or finished, timer is stale

    thread, result_q, timer, is_cancelled_flag = _worker_threads[bounds_name]

    try:
        # Check the queue without blocking
        result_data = result_q.get_nowait()
    except queue.Empty:
        return 0.1 # No result yet, poll again in 0.1 seconds

    # --- Result is ready, clean up worker state immediately ---
    _worker_threads.pop(bounds_name)
    is_cancelled_flag.set()
    if timer and bpy.app.timers.is_registered(timer):
        bpy.app.timers.unregister(timer)
    
    context = bpy.context
    scene = getattr(context, 'scene', None)
    if not scene:
        _updates_pending[bounds_name] = False
        return None

    bounds_obj = scene.objects.get(bounds_name)
    if not bounds_obj:
        _updates_pending[bounds_name] = False
        return None
        
    mesh_update_successful = False
    mesh_generation_error = False
    result_obj = None

    try:
        if result_data is None: # Worker was cancelled
            raise InterruptedError("Update was cancelled by a newer request.")
        if isinstance(result_data, Exception): # Worker had an error
            raise result_data

        mesh_data, meshing_time, actual_rendered_div = result_data
        sdf_settings_from_bounds = trigger_state.get('scene_settings')
        result_name = bounds_obj.get(constants.SDF_RESULT_OBJ_NAME_PROP)

        # Update current div to the one that was just rendered
        _current_divs[bounds_name] = actual_rendered_div

        # Adaptive div logic (only for viewport updates)
        if is_viewport_update:
            target_div = _target_divs.get(bounds_name, MIN_DIV)
            if meshing_time < 0.05: # Very fast, decrease target div (higher resolution)
                _target_divs[bounds_name] = max(MIN_DIV, target_div - 1)
            elif meshing_time > 0.5: # Slow, increase target div (lower resolution)
                _target_divs[bounds_name] = min(MAX_DIV, target_div + 1)

        # Find or create result object
        result_obj = utils.find_result_object(context, result_name)
        if not result_obj and result_name and sdf_settings_from_bounds.get("sdf_create_result_object"):
             new_mesh_bdata = bpy.data.meshes.new(name=result_name + "_Mesh")
             result_obj = bpy.data.objects.new(result_name, new_mesh_bdata)
             link_collection = bounds_obj.users_collection[0] if bounds_obj.users_collection else scene.collection
             link_collection.objects.link(result_obj)
             result_obj.matrix_world = Matrix.Identity(4) 
             result_obj.hide_select = True
        
        if result_obj and result_obj.type == 'MESH':
            # Update the existing mesh datablock
            new_mesh_bdata = result_obj.data
            new_mesh_bdata.clear_geometry() # Clear existing data

            if mesh_data and mesh_data[0]:
                new_mesh_bdata.from_pydata(mesh_data[0], [], mesh_data[1])
            new_mesh_bdata.update()
            mesh_update_successful = True
            
            # Manage SDF-related color attribute only on the final, high-resolution mesh
            is_final_update = not is_viewport_update or _current_divs.get(bounds_name, MAX_DIV) <= _target_divs.get(bounds_name, MIN_DIV)
            if mesh_update_successful and is_final_update:
                update_id = time.time_ns() % (2**30) # Unique ID for this update, kept within C int limits
                result_obj[constants.SDF_UPDATE_ID_PROP] = update_id
                print(f"FieldForge DEBUG: Starting color attribute calculation for {result_obj.name} (Update ID: {update_id})")
                _manage_sdf_color_attribute(result_obj, bounds_obj, context, update_id)
                _ensure_sdf_material(result_obj)

            if mesh_update_successful and len(new_mesh_bdata.polygons) > 0:
                 for poly in new_mesh_bdata.polygons:
                     poly.use_smooth = True
                 
                 # Auto Smooth Angle via Modifier for robustness
                 addon_modifier_name = "FieldForge_Smooth"
                 auto_smooth_angle_deg = sdf_settings_from_bounds.get("sdf_result_auto_smooth_angle", 45.0)
                 auto_smooth_angle_rad = math.radians(auto_smooth_angle_deg)
                 
                 existing_mod = result_obj.modifiers.get(addon_modifier_name)
                 if not existing_mod:
                     try:
                        with context.temp_override(object=result_obj, active_object=result_obj, selected_objects=[result_obj]):
                            bpy.ops.object.modifier_add_node_group(
                                asset_library_type='ESSENTIALS', asset_library_identifier="Essentials", 
                                relative_asset_identifier="geometry_nodes/smooth_by_angle.blend/NodeTree/Smooth by Angle")
                        existing_mod = result_obj.modifiers[-1]
                        existing_mod.name = addon_modifier_name
                     except Exception as e_mod:
                         print(f"FF WARN: Could not add 'Smooth by Angle' node group from asset library: {e_mod}")
                 
                 if existing_mod and existing_mod.node_group and "Input_1" in existing_mod:
                     try: existing_mod["Input_1"] = auto_smooth_angle_rad
                     except Exception: pass

    except Exception as e:
        mesh_generation_error = True
        mesh_update_successful = False
        if not isinstance(e, InterruptedError):
            print(f"FieldForge ERROR: Failed to apply mesh data for {bounds_name}: {e}")
        if result_obj and result_obj.data:
            try: result_obj.data.clear_geometry(); result_obj.data.update()
            except Exception: pass
    finally:
        if not mesh_generation_error and mesh_update_successful:
            update_sdf_cache(trigger_state, bounds_name)
        
        if is_viewport_update and _current_divs.get(bounds_name, MAX_DIV) > _target_divs.get(bounds_name, MIN_DIV):
            bounds_obj = scene.objects.get(bounds_name)
            if bounds_obj and utils.get_bounds_setting(bounds_obj, "sdf_auto_update"):
                bpy.app.timers.register(
                    lambda: run_sdf_update(bounds_name, trigger_state, is_viewport_update),
                    first_interval=0.01
                )
            else:
                _updates_pending[bounds_name] = False
        elif not is_viewport_update:
            # This was a manual update, so we don't schedule another one.
            # The color worker will set the pending flag to false when it's done.
            pass
        else:
            _updates_pending[bounds_name] = False

    return None

def run_sdf_update(bounds_name: str, trigger_state: dict, is_viewport_update: bool = False):
    """
    STARTS the threaded SDF generation and mesh update process.
    """
    global _updates_pending
    if not _lf_imported_ok:
        return

    _updates_pending[bounds_name] = True
    context = bpy.context
    if not context or not context.scene:
        if bounds_name in _updates_pending: _updates_pending[bounds_name] = False;
        return

    scene = context.scene
    bounds_obj = scene.objects.get(bounds_name)
    if not bounds_obj:
        if bounds_name in _updates_pending: _updates_pending[bounds_name] = False;
        return

    _cancel_and_clear_worker(bounds_name)
    
    sdf_settings = trigger_state.get('scene_settings')
    final_combined_shape = sdf_logic.process_sdf_hierarchy(bounds_obj, sdf_settings)

    if final_combined_shape is None:
        final_combined_shape = lf.emptiness()

    bounds_matrix = trigger_state.get('bounds_matrix')
    local_corners = [Vector(c) for c in ((-1,-1,-1), (1,-1,-1), (-1,1,-1), (1,1,-1), (-1,-1,1), (1,-1,1), (-1,1,1), (1,1,1))]
    world_corners = [(bounds_matrix @ c.to_4d()).xyz for c in local_corners]

    xyz_min = (min(c.x for c in world_corners), min(c.y for c in world_corners), min(c.z for c in world_corners))
    xyz_max = (max(c.x for c in world_corners), max(c.y for c in world_corners), max(c.z for c in world_corners))
    
    if bounds_name not in _current_divs or bounds_name not in _target_divs or _current_divs[bounds_name] <= _target_divs[bounds_name]:
        if is_viewport_update:
            _current_divs[bounds_name] = MAX_DIV
            _target_divs[bounds_name] = MIN_DIV
        else:
            _current_divs[bounds_name] = MIN_DIV
            _target_divs[bounds_name] = MIN_DIV
    
    elif is_viewport_update and _current_divs[bounds_name] > _target_divs[bounds_name]:
        _current_divs[bounds_name] -= 1

    div_for_worker = _current_divs[bounds_name]
    base_resolution_setting = sdf_settings.get("sdf_viewport_resolution" if is_viewport_update else "sdf_final_resolution", 10)

    result_q = queue.Queue()
    is_cancelled_flag = threading.Event()
    
    worker_thread = threading.Thread(
        target=_mesh_generation_worker,
        args=(result_q, is_cancelled_flag, final_combined_shape, bounds_name, base_resolution_setting, div_for_worker, xyz_min, xyz_max)
    )
    
    result_timer = bpy.app.timers.register(
        lambda: _apply_mesh_data_from_worker(bounds_name, trigger_state, is_viewport_update),
        first_interval=0.1
    )

    _worker_threads[bounds_name] = (worker_thread, result_q, result_timer, is_cancelled_flag)
    worker_thread.start()

def _cancel_and_clear_worker(bounds_name: str):
    """Safely cancels and cleans up a worker thread and its timer."""
    global _worker_threads, _color_worker_threads
    
    # Cancel mesh generation worker
    if bounds_name in _worker_threads:
        thread, _, timer, is_cancelled_flag = _worker_threads.pop(bounds_name)
        is_cancelled_flag.set() # Signal the thread to stop
        if timer and bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)

    # Cancel color calculation worker
    bounds_obj = bpy.data.objects.get(bounds_name)
    if bounds_obj:
        result_obj_name = bounds_obj.get(constants.SDF_RESULT_OBJ_NAME_PROP)
        if result_obj_name and result_obj_name in _color_worker_threads:
            thread, _, timer, is_cancelled_flag = _color_worker_threads.pop(result_obj_name)
            is_cancelled_flag.set() # Signal the thread to stop
            if timer and bpy.app.timers.is_registered(timer):
                bpy.app.timers.unregister(timer)

# --- Scene Update Handler (Dependency Graph) ---
@bpy.app.handlers.persistent
def ff_depsgraph_handler(scene, depsgraph):
    """ Blender dependency graph handler, called after updates. """
    if not _lf_imported_ok: return

    context = bpy.context
    if not context or not context.window_manager or not context.window_manager.windows: return
    if bpy.app.background: return
    screen = getattr(context, 'screen', None)
    if screen and getattr(screen, 'is_scrubbing', False): return

    if depsgraph is None or not hasattr(depsgraph, 'updates'): return

    bounds_to_recheck = set()

    for update in depsgraph.updates:
        updated_obj = getattr(update, 'id', None)
        if not isinstance(updated_obj, bpy.types.Object):
            continue

        try:
            evaluated_obj = updated_obj.evaluated_get(depsgraph) if depsgraph else updated_obj
        except (ReferenceError, AttributeError): 
            continue
        if not evaluated_obj: continue

        root_bounds = utils.find_parent_bounds(updated_obj)
        if root_bounds:
            bounds_to_recheck.add(root_bounds.name)
        elif updated_obj.get(constants.SDF_BOUNDS_MARKER, False):
            bounds_to_recheck.add(updated_obj.name)
        
        for dependent_bounds_name in state.get_dependent_bounds_for_linked_object(updated_obj.name):
            bounds_to_recheck.add(dependent_bounds_name)

    if bounds_to_recheck:
        for bounds_name in bounds_to_recheck:
            bpy.app.timers.register(
                lambda name_arg=bounds_name: check_and_trigger_update(name_arg, "depsgraph_or_link_event"),
                first_interval=0.0
            )

# --- Initial Update Check on Load ---
def initial_update_check_all():
    """ Schedules an initial state check for all existing Bounds objects. """
    context = bpy.context
    if not context or not context.scene: return None
    if not _lf_imported_ok: return None

    for bounds_obj in utils.get_all_bounds_objects(context):
        try:
            check_and_trigger_update(bounds_obj.name, "initial_check")
        except Exception as e:
            print(f"FieldForge ERROR: Failed initial check for {bounds_obj.name}: {e}")

    return None

# --- Cleanup Function ---
def clear_timers_and_state():
    """Cancels all active timers and clears global state dictionaries."""
    global _updates_pending, _sdf_update_caches, _worker_threads, _current_divs, _target_divs, _color_worker_threads
    for bounds_name in list(_worker_threads.keys()):
        _cancel_and_clear_worker(bounds_name)
    for result_name in list(_color_worker_threads.keys()):
        if result_name in _color_worker_threads:
            _, _, timer, is_cancelled_flag = _color_worker_threads.pop(result_name)
            is_cancelled_flag.set()
            if timer and bpy.app.timers.is_registered(timer):
                bpy.app.timers.unregister(timer)

    _updates_pending.clear()
    _sdf_update_caches.clear()
    _worker_threads.clear()
    _current_divs.clear()
    _target_divs.clear()
    _color_worker_threads.clear()
    clear_link_caches()