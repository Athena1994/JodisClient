import { AfterViewInit, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { MatListModule, MatSelectionList, MatSelectionListChange } from '@angular/material/list';
import { Job } from '../../models/job.interface';
import { JobService } from '../../services/jobs.service';
import { CommonModule } from '@angular/common';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { select, Store } from '@ngrx/store';
import { selectAssignedJobs, selectFinishedJobs, selectSelectedJob, selectUnassignedJobs } from '../../state/jobs/jobs.selectors';
import { jobsActions } from '../../state/jobs/jobs.actions';
import { MatCheckboxModule } from '@angular/material/checkbox';


@Component({
  selector: 'app-job-overview',
  standalone: true,
  imports: [MatListModule, CommonModule, MatCheckboxModule],
  templateUrl: './job-overview.component.html',
  styleUrl: './job-overview.component.css'
})
export class JobOverviewComponent implements OnInit {

    assignedJobs: Job[] = [];
    finishedJobs: Job[] = [];

    selectedJob$ = this.store.pipe(select(selectSelectedJob));

    // @ViewChild('unassignedJobsList', {static: true})
    // unassignedJobsList!: MatSelectionList;
    @ViewChild('assignedJobsList', {static: true})
    assignedJobsList!: MatSelectionList;
    @ViewChild('finishedJobsList', {static: true})
    finishedJobsList!: MatSelectionList;

    constructor(private store: Store) {
    }


    ngOnInit() {
      this.store.pipe(select(selectAssignedJobs)).subscribe((jobs) => {
          this.assignedJobs = jobs;
      });
      this.store.pipe(select(selectFinishedJobs)).subscribe((jobs) => {
          this.finishedJobs = jobs;
      });

      this.store.dispatch(jobsActions.load());
    }

    onJobSelected(job: Job) {
      this.store.dispatch(jobsActions.selectJob({ job }));
    }
}
