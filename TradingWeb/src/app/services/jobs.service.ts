import { Inject, Injectable} from "@angular/core";
import { Job } from "../models/job.interface";
import { HttpClient } from '@angular/common/http';
import { map, Observable, tap } from "rxjs";
import { API_URL_TOKEN } from "../app.config";

@Injectable({providedIn: 'root'})
export class JobService {

    constructor(private http: HttpClient,
                @Inject(API_URL_TOKEN) private api_url: string) {}

    getJobs(assigned: boolean,
            unassigned: boolean,
            finished: boolean): Observable<Job[]> {
        const params: any = {};
        if (assigned) {
            params.assigned = true;
        }
        if (unassigned) {
            params.unassigned = true;
        }
        if (finished) {
            params.finished = true;
        }

        return this.http.get<Job[]>(
            this.api_url.toString()+"jobs", {params});
    }
    createJob(name: string,
              description: string,
              config: JSON): Observable<Job> {
        return this.http.post<Job[]>(
            this.api_url.toString()+"job", ({
                name, description, config
            })).pipe(map(jobs => jobs[0]));
    }
    deleteJobs(jobs: Job[]): Observable<number[]> {
        return this.http.post<{deletedIds: number[]}>(
            this.api_url.toString()+"jobs/delete",
            ({ids: jobs.map(j => j.id)})).pipe(map(res => res.deletedIds));
    }
}