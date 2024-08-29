import { Component, ViewChild } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { ConfigValidationService } from '../../services/configValidation.service';
import { select, Store } from '@ngrx/store';
import { newJobActions } from '../../state/new-job/new-job.actions';
import { isPending, isReadyToCreate, selectDescription, selectError, selectName, selectStatus } from '../../state/new-job/new-job.selectors';
import { CommonModule } from '@angular/common';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { FormsModule } from '@angular/forms';
import { map } from 'rxjs';

@Component({
  selector: 'app-job-creator',
  standalone: true,
  imports: [MatFormFieldModule, MatInputModule, MatIconModule, CommonModule,
    MatProgressSpinnerModule, FormsModule
  ],
  templateUrl: './job-creator.component.html',
  styleUrl: './job-creator.component.css'
})
export class JobCreatorComponent {

  @ViewChild("nameField", {static: true})
  nameField!: MatFormFieldModule;

  @ViewChild("descriptionField", {static: true})
  descriptionField!: MatFormFieldModule;

  createEnabled$ = this.store.pipe(select(isReadyToCreate));
  error$ = this.store.pipe(select(selectError));
  status$ = this.store.pipe(select(selectStatus));

  name$ = this.store.pipe(select(selectName));
  description$ = this.store.pipe(select(selectDescription));

  configFile?: File;
  fileName?: string;

  showSpinner$ = this.store.pipe(select(isPending));
  spinnerText$ = this.store.pipe(select(selectStatus)).pipe(
    map(status => {
      switch (status) {
        case 'validating':
          return 'Validating config...';
        case 'creating':
          return 'Creating job...';
        default:
          return '';
      }
    })
  );

  onNameChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.store.dispatch(newJobActions.setName({ name: target.value }));
  }

  onDescriptionChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.store.dispatch(newJobActions.setDescription(
      { description: target.value }));
  }

  createJob() {
      this.store.dispatch(newJobActions.createJob());
  }

  async onFileSelected(event: any) {
    const file:File = event.target.files[0];

    if (file) {
        this.fileName = file.name;
        this.configFile = file;
        this.store.dispatch(newJobActions.validateConfig(
          { config: await file.text() }));
    }
  }

  constructor(private store: Store) {
  }


}
