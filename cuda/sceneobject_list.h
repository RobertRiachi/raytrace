#ifndef SCENEOBJECT_LIST_H
#define SCENEOBJECT_LIST_H

#include "sceneobject.h"

#include <vector>

class sceneobject_list: public sceneobject {
    public:
        size_t list_size;
        sceneobject **list;

        __device__ sceneobject_list() {}
        __device__ sceneobject_list(sceneobject **l, int n) {
            list_size = n;
            list = l;
        }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override;

};

__device__ bool sceneobject_list::get_bounding_box(bounding_box &output_box) const {
    if (list_size < 1) return false;

    bounding_box temp_box;
    bool first_box = true;

    for(int i = 0; i < list_size; i++) {
        if (!this->list[i]->get_bounding_box(temp_box)) return false;

        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;

}

__device__ bool sceneobject_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;

        }
    }
    return hit_anything;

}

#endif